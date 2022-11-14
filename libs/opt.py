import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("cls_type", type=str,default="one_class",choices=['one_class', 'multi_class', 'open_set'])
        self.parser.add_argument("--phase", type=str, default="train",choices=['train','test'])
        self.parser.add_argument("--cls_dataroot", type=str, default="../data")  
        self.parser.add_argument("--cls_dataset", type=str, default="out30")   
        self.parser.add_argument("--cls_network", type=str, default="hmm")
        self.parser.add_argument("--cls_imageSize", type=int)
        self.parser.add_argument("--cls_imageSize2", type=int, default=-1)
        self.parser.add_argument("--nc", type=int, default=3)
        self.parser.add_argument("--cls_batchsize", type=int, default=64)
        self.parser.add_argument("--metric", nargs="+", default=['AUC'])
        self.parser.add_argument("--metric_pos_label", type=str, default = '9')
        self.parser.add_argument("--metric_average", type=str)
        self.parser.add_argument("--outf", type=str, default='../output')
        self.parser.add_argument("--control_save_end", type=int, default=0, help='save the weights on terminal(default False)')
        self.parser.add_argument("--control_print", action='store_true', help='print the results on terminal(default False)')
        self.parser.add_argument("--control_save", action='store_true', help='save the results to files(default False)')
        self.parser.add_argument("--load_weights", action='store_true', help='load parameters(default False)')
        self.parser.add_argument("--gray_image", action='store_true', help='convert image to grayscale(default False)')
        self.parser.add_argument("--aug_methods", nargs='+', default=[])

        self.args = self.parser.parse_known_args()[0]
        self.unknown_args = self.parser.parse_known_args()[1]

        if self.args.gray_image or self.args.cls_dataset == 'mnist':
            self.change_args("nc",1)

        if self.args.phase == 'test':
            self.change_args("control_save_end",0)

    def parse(self):
        self.args = self.parser.parse_known_args()[0]
        self.unknown_args = self.parser.parse_known_args()[1]

    def add_args(self, arg_pairs):
        if not arg_pairs is None:
            for arg_name, arg_value in zip(arg_pairs.keys(), arg_pairs.values()):
                if arg_name in self.unknown_args:
                    if len(self.unknown_args[self.unknown_args.index(arg_name):]) > 2:
                        if not '--' in self.unknown_args[self.unknown_args.index(arg_name)+2]:
                            self.parser.add_argument(arg_name, nargs="+", default = arg_value)
                            continue
                    elif arg_name in ['--known_lbl']:
                        self.parser.add_argument(arg_name, type = type(arg_value), nargs ="+", default = arg_value)
                        continue
                self.parser.add_argument(arg_name, type = type(arg_value), default = arg_value)
            self.parse()

    def change_args(self, name, value):
        exec("self.parser.set_defaults({} = {})".format(name, value))
        self.parse()

    def get_args(self):
        return self.args

    def get_unknown_args(self):
        return self.unknown_args