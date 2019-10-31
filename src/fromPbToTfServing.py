import argparse
import sys

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.gfile import GFile
from tensorflow import GraphDef, Graph, import_graph_def, Session


def main(args):
    with GFile(args.frozen_model_path, "rb") as f:
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())

    with Session() as sess:
        # Then, we import the graph_def into a new Graph and returns it
        with Graph().as_default() as graph:
            import_graph_def(graph_def, name='')
            signature = predict_signature_def(
                inputs={'image_batch': graph.get_tensor_by_name('image_batch:0'),
                        'phase_train': graph.get_tensor_by_name('phase_train:0')},
                outputs={'embeddings': graph.get_tensor_by_name('embeddings:0')}
            )

            builder = saved_model_builder.SavedModelBuilder(args.output_model_dir)
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tag_constants.SERVING],
                signature_def_map={'serving_default': signature}
            )
            builder.save()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_model_path', type=str, help='Frozen model path.')
    parser.add_argument('--output_model_dir', type=str, help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))