import os
import csv
class Visualizer():
    def __init__(self,parser):
        super(Visualizer, self).__init__()
        self.args = parser.get_args()
        self.save_mode = {'matrix':'txt',
            'curve':'csv'}
        self.non_value_metrics_name = []
        self.args.outf = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.cls_dataset)
        dir_name = ''
        if self.args.control_save:
            if self.args.cls_type  in ["one_class", "open_set"]:
                for i in range(len(self.args.known_lbl)):
                    dir_name += self.args.known_lbl[i] if i == len(self.args.known_lbl)-1 else self.args.known_lbl[i]+'_'
                    self.outf = os.path.join(self.args.outf, self.args.cls_type, self.args.cls_network, dir_name)
            else:
                self.outf = os.path.join(self.args.outf, self.args.cls_type, self.args.cls_network)
            if not os.path.exists(self.outf):
                os.makedirs(self.outf)

    def plot_csv(self, file_name, message):
        csv_menu = os.path.join(self.outf, file_name)
        with open(csv_menu, "a", newline='') as file:
            csv_file = csv.writer(file)
            csv_file.writerows(message)

    def plot_txt(self, file_name, message):
        txt_menu = os.path.join(self.outf, file_name)
        with open(txt_menu, 'a', newline='\n') as file:
            file.write('%s\n' % message)

    def plot_menu(self, best):
        menu_path = self.args.outf +'/menu.txt'
        menu = os.path.join(menu_path)
        with open(menu, "a",newline='\n') as file:
            if self.class_type  in ["one_class", "open_set"]:
                datas = [self.args.cls_network, self.args.known_lbl, best]
            if self.args.cls_type == 'multi_class':
                datas = [self.args.cls_network, best]
            file.write('%s\n' % datas)

    def loggin(self, metric_cluster, current_epoch, best_value, indicator_for_best):
        self.metric_properties = metric_cluster.metric_properties
        self.file_name = ''
        self.message = '''Epoch : {}, {}'s '''.format(current_epoch, self.args.cls_network)
        for i in range(len(self.args.metric)):
            if not metric_cluster.values[i] is None:
                self.message += '''{} is {:.4f} '''.format(self.args.metric[i], metric_cluster.values[i])
                self.message += ''' the best {} is {:.4f} '''.format(self.args.metric[i], best_value) if i == indicator_for_best else ''
                # if i == indicator_for_best:
                #     self.message += '''{} is {:.4f}, the best {} is {:.4f} '''.format(self.args.metric[i], metric_cluster.values[i], self.args.metric[i], best_value)
                # else:
                #     self.message += '''{} is {:.4f} '''.format(self.args.metric[i], metric_cluster.values[i])
                self.file_name += self.args.metric[i] if i == indicator_for_best else '_'+self.args.metric[i]
            else:
                if self.metric_properties[self.args.metric[i]] == 'matrix':
                    exec("self.high_dim_metric_value = metric_cluster.{}".format(self.args.metric[i]))
                    temp = 'Epoch:'+str(current_epoch)+'\n'+str(self.high_dim_metric_value)
                    exec("self.{}_message = temp ".format(self.args.metric[i]))
                    # exec("self.{}_name = {}.txt".format(self.args.metric[i], self.args.metric[i]))
                elif self.metric_properties[self.args.metric[i]] == 'curve':
                    exec("self.{}_message = metric_cluster.{}".format(self.args.metric[i], self.args.metric[i]))
                temp = self.args.metric[i] + '.' + self.save_mode[self.metric_properties[self.args.metric[i]]]
                exec("self.{}_name = temp".format(self.args.metric[i]))
                if current_epoch == 1 or current_epoch == 'test': self.non_value_metrics_name.append(self.args.metric[i])
        self.file_name += '_epoch{}.txt'.format(self.args.cls_epochs)

    def save(self):
        self.plot_txt(self.file_name, self.message)
        for i in range(len(self.non_value_metrics_name)):
            exec("self.temp_message = self.{}_message".format(self.non_value_metrics_name[i]))
            exec("self.temp_name = self.{}_name".format(self.non_value_metrics_name[i]))
            exec("self.plot_{}(self.temp_name, self.temp_message)".format(self.save_mode[self.metric_properties[self.non_value_metrics_name[i]]]))
            # exec("self.{}_name = {}.txt".format(self.args.metric[i], self.args.metric[i]))

    def output(self):
        print(self.message)
        pass

    def plot_inference_time(self, time):
        message = '''{}'s inference_time: {} ms'''.format(self.args.cls_network, time)
        if self.args.control_print:
            print(message)
        if self.args.control_save:
            self.file_name = 'inference_time.txt'
            self.plot_txt(self.file_name, message)
        # return message

    def plot_fps(self, fps):
        message = '''{}'s fps: {} img/s'''.format(self.args.cls_network, fps)
        if self.args.control_print:
            print(message)
        if self.args.control_save:
            self.file_name = 'fps.txt'
            self.plot_txt(self.file_name, message)
        # return message

    def plot_macs_params(self, macs, params, net_name):
        message = '''{} : macs: {}, params: {}.'''.format(net_name, macs, params)
        if self.args.control_print:
            print(message)
        if self.args.control_save:
            self.file_name = 'macs_params.txt'
            self.plot_txt(self.file_name, message)
        # return message
