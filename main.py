from libs.opt import options
from libs.data import data

opt = options()

exec("from {}.{}.model import {}".format(opt.get_args().cls_type, opt.get_args().cls_network, opt.get_args().cls_network))

exec("classifier = {}(opt)".format(opt.get_args().cls_network))

# classifier.get_inference_time()
# classifier.get_fps()

data = data(opt, classifier.is_deep_model)

assert opt.unknown_args == [], 'Existing unknown unparsed arguments.'

train_loader, test_loader = data.get_data()

# classifier.addvis(Visualizer(opt))

exec("classifier.{}(train_loader, test_loader)".format(opt.get_args().phase))



# one_classifier_set = {"ALOCC":ALOCC, 
#                         "AuthGAN":AuthGAN, 
#                         "AnoGAN":AnoGAN, 
#                         "f_AnoGAN":f_AnoGAN,
#                         "Ganomaly":Ganomaly, 
#                         "OCGAN":OCGAN, 
#                         "DGN":DGN}

# multi_classifier_set = {"mnistcnn":mnistcnn, 
#                         "cifarcnn":cifarcnn, 
#                         "efm":efm, 
#                         "mifs":mifs, 
#                         "dfm":dfm, 
#                         "emd":emd, 
#                         "envelope_den":envelope_den, 
#                         "hmm":hmm, 
#                         "knn_svm":knn_svm, 
#                         "omp_kmeans_nn":omp_kmeans_nn,
#                         "mfrnet":mfrnet, 
#                         "densenet":densenet, 
#                         "emcnet":emcnet, 
#                         "sann":sann, "mnet":mnet, 
#                         "mrmdcnn":mrmdcnn, 
#                         "pyalexnet":pyalexnet}

# open_classifier_set = {"opengan":opengan}



