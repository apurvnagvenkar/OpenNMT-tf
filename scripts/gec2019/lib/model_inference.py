
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2





class Inference(object):

    def __init__(self, nmt_stub, channel, model_name, timeout):
        """

        :param nmt_stub:
        :param channel:
        :param timeout:
        """
        self.nmt_stub = nmt_stub
        self.channel = channel
        self.timeout = timeout
        self.model_name = model_name

    def infer(self, input):
        """

        :param input: sentence piece tokens
        :return:
        """
        output = self.correction(stub=self.nmt_stub, model_name=self.model_name, tokens=input, timeout=self.timeout)
        zip_correction = self.parse_correction_result(result=output.result())
        return list(zip_correction)

    def correction(self, stub, model_name, tokens, timeout=5.0):
        """corrects a sequence of tokens.
      Args:
        stub: The prediction service stub.
        model_name: The model to request.
        tokens: A list of tokens.
        timeout: Timeout after this many seconds.
      Returns:
        A future.
      """
        length = len(tokens)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.inputs["tokens"].CopyFrom(
            tf.make_tensor_proto([tokens], shape=(1, length)))
        request.inputs["length"].CopyFrom(
            tf.make_tensor_proto([length], shape=(1,)))

        return stub.Predict.future(request, timeout)

    def parse_correction_result(self, result):
        """
        Parses a correction result.

        :param result: A list of hypotheses along with length and score.
        :return: zip output containing hypothesis, length and log probability
        """
        lengths = tf.make_ndarray(result.outputs["length"])[0]
        log_probs = tf.make_ndarray(result.outputs["log_probs"])[0]
        hypotheses = tf.make_ndarray(result.outputs["tokens"])[0]
        zip_correction = zip(hypotheses, lengths, log_probs)
        return zip_correction

