import numpy as np

from multiprocessing import Pipe, connection
from threading import Thread
from time import time

from logging import getLogger

from reversi_zero.agent.model import ReversiModel
from reversi_zero.config import Config

from reversi_zero.lib.model_helpler import reload_newest_next_generation_model_if_changed, load_best_model_weight, \
    save_as_best_model, reload_best_model_weight_if_changed
import tensorflow as tf
import os
import keras.backend as K



logger = getLogger(__name__)


class ReversiModelAPI:
    def __init__(self, config: Config, agent_model):
        """

        :param config:
        :param reversi_zero.agent.model.ReversiModel agent_model:
        """
        self.config = config
        self.agent_model = agent_model

    def predict(self, x):
        assert x.ndim in (3, 4)
        assert x.shape == (2, 15, 15) or x.shape[1:] == (2, 15, 15)
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 2, 15, 15)

        policy, value = self._do_predict(x)

        if orig_x.ndim == 3:
            return policy[0], value[0]
        else:
            return policy, value

    def _do_predict(self, x):
        return self.agent_model.model.predict_on_batch(x)


class MultiProcessReversiModelAPIServer:
    # https://github.com/Akababa/Chess-Zero/blob/nohistory/src/chess_zero/agent/api_chess.py

    def __init__(self, config: Config):
        """

        :param config:
        """
        self.config = config
        self.model = None  # type: ReversiModel
        self.model_list = None
        self.gpu_num = 0
        self.connections = []

    def get_api_client(self):
        me, you = Pipe()
        self.connections.append(me)
        return MultiProcessReversiModelAPIClient(self.config, None, you)

    def start_serve(self,num=0):
        self.gpu_num = num
        # with tf.device('/gpu:' + str(self.gpu_num)):
        #     self.model = self.load_model()
        self.model_list = self.load_model_list()
        # threading workaround: https://github.com/keras-team/keras/issues/5640
        for model in self.model_list:
            model.model._make_predict_function()
        self.graph = tf.get_default_graph()

        prediction_worker = Thread(target=self.prediction_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def ready_to_serve(self,num):
        self.gpu_num = num
        with tf.device('/gpu:' + str(self.gpu_num)):
            self.model = self.load_model()
        self.model_list = self.load_model_list()
        self.model.model._make_predict_function()
        self.graph = tf.get_default_graph()

        prediction_worker = Thread(target=self.prediction_worker, name="prediction_worker")
        prediction_worker.daemon = True
        return prediction_worker

    def prediction_worker(self):
        logger.debug("prediction_worker started")
        average_prediction_size = []
        last_model_check_time = time()
        count = 0
        while True:
            if last_model_check_time+300 < time():
                # with tf.device('/gpu:' + str(self.gpu_num)):
                # self.try_reload_model()
                self.try_reload_model_list()
                last_model_check_time = time()
                logger.debug(f"average_prediction_size={np.average(average_prediction_size)}")
                average_prediction_size = []
            ready_conns = connection.wait(self.connections, timeout=0.001)  # type: list[Connection]
            if not ready_conns:
                continue
            data = []
            size_list = []
            for conn in ready_conns:
                x = conn.recv()
                data.append(x)  # shape: (k, 2, 15, 15)
                size_list.append(x.shape[0])  # save k
            average_prediction_size.append(np.sum(size_list))
            array = np.concatenate(data, axis=0)
            # print(f'Now use GPU#{count}')
            policy_ary, value_ary = self.model_list[count].model.predict_on_batch(array)
            count += 1
            count %= self.config.model.num_gpus
            idx = 0
            for conn, s in zip(ready_conns, size_list):
                conn.send((policy_ary[idx:idx+s], value_ary[idx:idx+s]))
                idx += s

    def load_model(self):
        from reversi_zero.agent.model import ReversiModel
        model = ReversiModel(self.config)
        loaded = False
        if not self.config.opts.new:
            if self.config.play.use_newest_next_generation_model:
                loaded = reload_newest_next_generation_model_if_changed(model) or load_best_model_weight(model)
            else:
                loaded = load_best_model_weight(model) or reload_newest_next_generation_model_if_changed(model)

        if not loaded:
            model.build()
            save_as_best_model(model)
        return model

    def try_reload_model(self):
        try:
            logger.debug("check model")
            if self.config.play.use_newest_next_generation_model:
                with tf.device('/cpu:0'):
                    reload_newest_next_generation_model_if_changed(self.model, clear_session=True)
                with tf.device('/gpu:' + str(self.gpu_num)):
                    self.model = self.model
            else:
                reload_best_model_weight_if_changed(self.model, clear_session=True)
        except Exception as e:
            logger.error(e)

    def load_model_list(self):
        model_list = []
        from reversi_zero.agent.model import ReversiModel
        for i in range(self.config.model.num_gpus):
            with tf.device('/gpu:' + str(i)):
                model = ReversiModel(self.config)
                loaded = False
                if not self.config.opts.new:
                    if self.config.play.use_newest_next_generation_model:
                        loaded = reload_newest_next_generation_model_if_changed(model) or load_best_model_weight(model)
                    else:
                        loaded = load_best_model_weight(model) or reload_newest_next_generation_model_if_changed(model)

                if not loaded:
                    model.build()
                    save_as_best_model(model)
                model_list.append(model)
            print(f'Create model on GPU#{i}')

        return model_list

    def try_reload_model_list(self):
        try:
            logger.debug("check model list")
            from reversi_zero.lib.data_helper import get_next_generation_model_dirs
            rc = self.config.resource
            dirs = get_next_generation_model_dirs(rc)
            if not dirs:
                logger.debug("No next generation model exists.")
                return False
            model_dir = dirs[-1]
            weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
            digest = self.model_list[0].fetch_digest(weight_path)
            if digest and digest != self.model_list[0].digest:
                K.clear_session()
                del self.model_list
                self.model_list = self.load_model_list()
                for model in self.model_list:
                    model.model._make_predict_function()
            # for i in range(self.config.model.num_gpus):
            #     os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            #     with tf.device('/gpu:' + str(i)):
            #         if self.config.play.use_newest_next_generation_model:
            #             reload_newest_next_generation_model_if_changed(self.model_list[i], clear_session=True,config=self.config)
            #         else:
            #             reload_best_model_weight_if_changed(self.model_list[i], clear_session=True)
        except Exception as e:
            logger.error(e)

class MultiProcessReversiModelAPIClient(ReversiModelAPI):
    def __init__(self, config: Config, agent_model, conn):
        """

        :param config:
        :param reversi_zero.agent.model.ReversiModel agent_model:
        :param Connection conn:
        """
        super().__init__(config, agent_model)
        self.connection = conn

    def _do_predict(self, x):
        self.connection.send(x)
        return self.connection.recv()
