import kenlm
import sentencepiece as spm
import grpc
import simplediff
import re

from lib.model_inference import Inference
from tensorflow_serving.apis import prediction_service_pb2_grpc

sp = spm.SentencePieceProcessor()
sp.Load("/home/versionx/Documents/gec2019/models/sent_piece/bea19.train.sent_piece.model")

host = "93.115.28.54"
#host="http://34.73.193.69"
port = 9100
model_name = "gec"

channel = grpc.insecure_channel("%s:%d" % (host, port))
nmt_stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
timeout = 1000

def generate_hypothesis(line):
    """

    :param line:
    :param lm:
    :return:
    """
    hypotheses = []
    sp_encode = sp.EncodeAsPieces(line)
    inf = Inference(nmt_stub, channel, model_name, timeout)
    output_list = inf.infer(sp_encode)
    for output in output_list:
       # print(output[0], output[1], output[2])
        decoded_output = sp.DecodePieces(list(output[0]))
        decoded_output = decoded_output.decode("utf-8").replace("<blank>", "")
        hypotheses.append([decoded_output, output[2]])
    return hypotheses



def generate(input_file, output_file):
    f_input = open(input_file, "r")
    f_write = open(output_file, "w")
    count = 0
    for line in f_input.readlines():
        hypotheses = generate_hypothesis(line=line)
        for hypothesis in hypotheses:
            f_write.write(str(count) +" ||| " + hypothesis[0] +" ||| "+ str(hypothesis[1]) +"\n")
        count +=1
    f_write.close()

input_file = "/home/versionx/Documents/crimson_mt_gc/evaluation/data/conll2014test/original.txt"
output_file = "/home/versionx/Documents/gec2019/output/bea19.transformerbig_v5_beam12/nbest/conll2014test.nbest.out"
generate(input_file=input_file, output_file=output_file)