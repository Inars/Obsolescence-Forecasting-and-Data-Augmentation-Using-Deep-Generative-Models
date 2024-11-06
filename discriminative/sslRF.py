
import os
import numpy as np
import multiprocessing
import pandas as pd
from pandas import read_csv
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
from sdv.metadata import SingleTableMetadata
import random
import logging


def f1(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = y_pred.argmax(axis=1)
    out = f1_score(y_real, y_pred, average="macro")
    logging.debug("F1 MACRO: " + str(out))
    return out


def f1W(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = y_pred.argmax(axis=1)
    out = f1_score(y_real, y_pred, average="weighted")
    logging.debug("F1 WEIGHTED: " + str(out))
    return out


def auc(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    if len(np.unique(y_real)) == 2:
        y_pred = y_pred[:, 1]
    out = roc_auc_score(y_real, y_pred, multi_class="ovr", average="macro")
    logging.debug("AUC MACRO: " + str(out))
    return out


def aucW(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    if len(np.unique(y_real)) == 2:
        y_pred = y_pred[:, 1]
    out = roc_auc_score(y_real, y_pred, multi_class="ovr", average="weighted")
    logging.debug("AUC WEIGHTED: " + str(out))
    return out 


def accuracy(y_real, y_pred):
    y_real = np.array(y_real)
    y_pred = y_pred.argmax(axis=1)
    out = sum(y_real == y_pred) / len(y_real)
    baseline = sum(y_real == np.random.choice(y_real, size=len(y_real), replace=True)) / len(y_real)

    logging.debug("ACCURACY: " + str(round(out, 4)) + " | Baseline: " + str(round(baseline, 4)))
 
    return out


def iforest(X, num_estimators=100, random_state=1102, contamination=0.05):

    clf = IsolationForest(
        n_estimators=num_estimators,
        n_jobs=int(np.max([multiprocessing.cpu_count()-2, 1])), 
        random_state=random_state, 
        max_features=np.max([int(0.3 * X.shape[1]), 1]), 
        contamination=contamination

    )
    array_inliers = np.array(clf.fit_predict(X)) > 0

    return array_inliers


class ModelRF:


    def __init__(self):
        np.random.seed(1102)
        self.model = RandomForestClassifier(
            n_estimators=25,
            max_depth=4, 
            random_state=1102, 
            n_jobs=int(np.max([multiprocessing.cpu_count()-2, 1]))
        )
        self.name = "RANDOM-FOREST"


    def fit(self, X, y, Xu=None):
        np.random.seed(1102)
        self.model.fit(X, y)

    
    def predict(self, X):
        np.random.seed(1102)
        return self.model.predict_proba(X)


class ModelKMeansRF:


    def __init__(self):
        np.random.seed(1102)
        self.model = RandomForestClassifier(
            max_depth=5, random_state=1102, 
            n_jobs=int(np.max([multiprocessing.cpu_count()-2, 1]))
        )

        self.name = "KMEANS-RF"


    def fit(self, X, y, Xu):

        np.random.seed(1102)

        Xtot = np.vstack((X, Xu))
        array_bool_labelled = np.append(np.repeat(True, X.shape[0]), np.repeat(False, Xu.shape[0]))
        ytot = np.append(np.array(y), np.repeat(-1, Xu.shape[0]))

        if Xu.shape[0] > 0:

            scaler = MinMaxScaler()
            Xtot_scaled = scaler.fit_transform(Xtot)

            model_kmeans = KMeans(n_clusters=int(Xtot.shape[0] / 30.0), random_state=1102)
            model_kmeans.fit(Xtot_scaled)
            labels_kmeans = np.array(model_kmeans.labels_)

            for k in np.unique(sorted(labels_kmeans)):
                obj_model = ModelRF()
                array_bool_l_tmp = array_bool_labelled & (labels_kmeans == k)
                array_bool_u_tmp = (~array_bool_labelled) & (labels_kmeans == k)

                if (sum(array_bool_l_tmp) > 0) and (sum(array_bool_u_tmp) > 0):

                    X_tmp = Xtot[array_bool_l_tmp, :]
                    y_tmp = ytot[array_bool_l_tmp]

                    if len(np.unique(y_tmp)) == 1:
                        ytot[array_bool_u_tmp] = y_tmp[0]
                    else:
                        obj_model.fit(X_tmp, y_tmp)
                        tmp_y_values = sorted(np.unique(y_tmp))
                        ytot[array_bool_u_tmp] = np.take(tmp_y_values, obj_model.predict(Xtot[array_bool_u_tmp, :]).argmax(axis=1))

        Xtot = Xtot[ytot >= 0, :]
        ytot = ytot[ytot >=0]

        obj_model = ModelRF()
        obj_model.fit(Xtot, ytot)

        self.model_rf = obj_model

    
    def predict(self, X):
        np.random.seed(1102)
        return self.model_rf.predict(X)
    

def augmented_data_model_run(
    obj_data, 
    class_model, 
    percentage_test, 
    dict_perf={
        "ACCURACY": accuracy, 
        "AUC_MACRO": auc, 
        "AUC_WEIGHTED": aucW, 
        "F1_MACRO": f1, 
        "F1_WEIGHTED": f1W
    },
    cv_folds=5,  
    verbose=True
):
    np.random.seed(1102)
    random.seed(1102)

    skf = StratifiedKFold(
        n_splits=cv_folds, 
        random_state=1102,
        shuffle=True
    )

    idx_unlabeled = obj_data.y_augmented == -1
    Xu = obj_data.X_augmented[idx_unlabeled, :]
    yu = obj_data.y_augmented[idx_unlabeled]
    Xl = obj_data.X_augmented[~idx_unlabeled, :]
    yl = obj_data.y_augmented[~idx_unlabeled]

    X = np.vstack((Xl, Xu))
    y = np.append(yl, yu)

    list_of_dicts = []
    list_of_models = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        obj_model = class_model()

        if verbose:
            logging.info(
                "MODEL " + obj_model.name + " " 
                + "DATA " + obj_data.name + " " 
                + "T" + str(percentage_test) + " " 
                + "FOLD " + str(i+1) + " "
            )
        
        idx_train_unlabeled = y[train_index] == -1
        idx_test_unlabeled = y[test_index] == -1
        X_labeled = X[train_index][~idx_train_unlabeled, :]
        y_labeled = y[train_index][~idx_train_unlabeled]
        X_unlabeled = np.vstack((X[train_index][idx_train_unlabeled, :], X[test_index][idx_test_unlabeled, :]))
        X_test = X[test_index][~idx_test_unlabeled, :]
        y_test = y[test_index][~idx_test_unlabeled]

        obj_model.fit(X_labeled, y_labeled, X_unlabeled)
        array_test_pred = obj_model.predict(X_test)
        array_test_real = y_test
        list_of_models.append(obj_model)

        row = {}
        for key_perf in dict_perf.keys():
            row[key_perf] = dict_perf[key_perf](array_test_real, array_test_pred)
        list_of_dicts.append(row)
    perf_df = pd.DataFrame(list_of_dicts)
    best_model = list_of_models[np.argmax(perf_df["ACCURACY"])]
    best_perf = perf_df.loc[np.argmax(perf_df["ACCURACY"]), :]

    temp_df_mean = perf_df.mean().to_frame().T
    temp_df_std = perf_df.std().to_frame().T
    perfs_df = pd.concat([temp_df_mean, temp_df_std], axis=0)

    if verbose: 
        logging.info('\t\n'+ perfs_df.to_string().replace('\n', '\n\t'))

    return perfs_df, best_model, best_perf


def encoded_data_model_run(
    obj_data, 
    class_model, 
    percentage_test, 
    dict_perf={
        "ACCURACY": accuracy, 
        "AUC_MACRO": auc, 
        "AUC_WEIGHTED": aucW, 
        "F1_MACRO": f1, 
        "F1_WEIGHTED": f1W
    },
    cv_folds=5,  
    verbose=True
):
    np.random.seed(1102)
    random.seed(1102)

    skf = StratifiedKFold(
        n_splits=cv_folds, 
        random_state=1102,
        shuffle=True
    )
    
    X = obj_data.X_encoded
    y = obj_data.y_encoded

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=percentage_test, 
        random_state=1102,
        stratify=y
    )

    list_of_dicts = []
    list_of_models = []
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        obj_model = class_model()

        if verbose:
            logging.info(
                "MODEL " + obj_model.name + " " 
                + "DATA " + obj_data.name + " " 
                + "T" + str(percentage_test) + " " 
                + "FOLD " + str(i+1) + " "
            )

        obj_model.fit(X_train[train_index], y_train[train_index])
        array_test_pred = obj_model.predict(X_train[test_index])
        array_test_real = y_train[test_index]
        list_of_models.append(obj_model)

        row = {}
        for key_perf in dict_perf.keys():
            row[key_perf] = dict_perf[key_perf](array_test_real, array_test_pred)
        list_of_dicts.append(row)
    perf_df = pd.DataFrame(list_of_dicts)
    best_model = list_of_models[np.argmax(perf_df["ACCURACY"])]
    best_perf = perf_df.loc[np.argmax(perf_df["ACCURACY"]), :]

    array_test_pred = obj_model.predict(X_test)
    array_test_real = y_test
    row = {}
    for key_perf in dict_perf.keys():
        row[key_perf] = dict_perf[key_perf](array_test_real, array_test_pred)
    best_perf_final = pd.DataFrame(row, index=[0])

    temp_df_mean = perf_df.mean().to_frame().T
    temp_df_std = perf_df.std().to_frame().T
    perfs_df = pd.concat([temp_df_mean, temp_df_std], axis=0)

    if verbose: 
        logging.info('\t\n'+ perfs_df.to_string().replace('\n', '\n\t'))

    return perfs_df, best_model, best_perf, best_perf_final


def data_model_run(
    obj_data, 
    class_model, 
    percentage_test, 
    dict_perf={
        "ACCURACY": accuracy, 
        "AUC_MACRO": auc, 
        "AUC_WEIGHTED": aucW, 
        "F1_MACRO": f1, 
        "F1_WEIGHTED": f1W
    },
    verbose=True
):
    np.random.seed(1102)
    random.seed(1102)
    
    X = obj_data.X
    y = obj_data.y

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=percentage_test, 
        random_state=1102,
        shuffle=True
    )

    obj_model = class_model()

    if verbose:
        logging.info(
            "MODEL " + obj_model.name + " " 
            + "DATA " + obj_data.name + " " 
            + "T" + str(percentage_test) + " " 
        )

    obj_model.fit(X_train, y_train)
    array_test_pred = obj_model.predict(X_train)
    array_test_real = y_train

    row = {}
    for key_perf in dict_perf.keys():
        row[key_perf] = dict_perf[key_perf](array_test_real, array_test_pred)
    perf_df = pd.DataFrame(row, index=[0])

    array_test_pred = obj_model.predict(X_test)
    array_test_real = y_test
    row = {}
    for key_perf in dict_perf.keys():
        row[key_perf] = dict_perf[key_perf](array_test_real, array_test_pred)
    best_perf_final = pd.DataFrame(row, index=[0])

    if verbose: 
        logging.info('\t\n'+ perf_df.to_string().replace('\n', '\n\t'))

    return perf_df, obj_model, best_perf_final


# main function
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    class Data:
        def __init__(self, name="moons", generator="CTGAN"): 
            self.name = name
            self.generator = generator
            self.path_original = "../data/" + self.name + ".csv"
            self.path_generated = "../data/generated/" + self.generator + "/" + self.name + ".csv"
            self.path_encoded = "../data/encoded/" + self.name + ".csv" if self.name != "moons" else "../data/" + self.name + ".csv"

        def load(self):
            dtf_data_original = read_csv(self.path_original, sep=",", encoding="latin1")
            dtf_data_encoded = read_csv(self.path_encoded, sep=",", encoding="latin1")
            dtf_data_generated = read_csv(self.path_generated, sep=",", encoding="latin1")
            dtf_data_generated["label"] = -1
            dtf_data = pd.concat([dtf_data_encoded, dtf_data_generated], axis=0)
            self.X = np.array(dtf_data_original.loc[:, dtf_data_original.columns != "label"])
            self.y = np.array(dtf_data_original["label"])
            self.X_encoded = np.array(dtf_data_encoded.loc[:, dtf_data_encoded.columns != "label"])
            self.y_encoded = np.array(dtf_data_encoded["label"])
            self.X_augmented = np.array(dtf_data.loc[:, dtf_data.columns != "label"])
            self.y_augmented = np.array(dtf_data["label"])
        
        def parse(self):
            array_bool_inliers = iforest(
                self.X, 
                num_estimators=100, 
                random_state=1102, 
                contamination=0.05
            )
            self.X = self.X[array_bool_inliers, :]
            self.y = self.y[array_bool_inliers]
            array_bool_inliers = iforest(
                self.X_encoded, 
                num_estimators=100, 
                random_state=1102, 
                contamination=0.05
            )
            self.X_encoded = self.X_encoded[array_bool_inliers, :]
            self.y_encoded = self.y_encoded[array_bool_inliers]
            array_bool_inliers = iforest(
                self.X_augmented, 
                num_estimators=100, 
                random_state=1102, 
                contamination=0.05
            )
            self.X_augmented = self.X_augmented[array_bool_inliers, :]
            self.y_augmented = self.y_augmented[array_bool_inliers]

    DATASET_NAMES = ['moons', 'arrow', 'phone']
    GENERATOR_NAMES = ['CTGAN', 'TVAE', 'NF']

    for DATASET_NAME in DATASET_NAMES:
        for GENERATOR_NAME in GENERATOR_NAMES:
            print("Dataset:", DATASET_NAME)
            print("Generator:", GENERATOR_NAME)

            obj_data = Data(name=DATASET_NAME, generator=GENERATOR_NAME)
            obj_data.load()
            obj_data.parse()

            # RF on augmented data
            print("Augmented data", DATASET_NAME)
            df_performance, model, df_perf_best = augmented_data_model_run(
                obj_data=obj_data, 
                class_model=ModelKMeansRF, 
                percentage_test=0.2, 
                dict_perf={
                    "ACCURACY": accuracy, 
                    "AUC_MACRO": auc, 
                    "AUC_WEIGHTED": aucW, 
                    "F1_MACRO": f1, 
                    "F1_WEIGHTED": f1W
                },
                cv_folds=5, 
                verbose=True
            )
            break
            array_pred = model.predict(obj_data.X_augmented)
            array_pred = array_pred.argmax(axis=1)
            obj_data.y_augmented = np.array(array_pred)

            df_data = pd.DataFrame(obj_data.X_augmented, columns=["x1", "x2"], dtype=float)
            df_data.insert(len(df_data.columns), "label", obj_data.y_augmented)

            df_data.to_csv("../data/discriminated/" + GENERATOR_NAME + "_" + DATASET_NAME + ".csv", sep=",", index=False)
            df_performance.to_csv("../output/RF/" + GENERATOR_NAME + "_" + DATASET_NAME + ".csv", sep=",", index=False)
            df_perf_best.to_csv("../output/RF/" + GENERATOR_NAME + "_" + DATASET_NAME + "_best.csv", sep=",", index=False)
    
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df_data)
            if not os.path.exists('../data/metadata/discriminated/' + DATASET_NAME + '.json'):
                metadata.save_to_json(filepath= '../data/metadata/discriminated/' + DATASET_NAME + '.json')

            # RF on encoded data
            print("Encoded data", DATASET_NAME)
            df_performance, _, df_perf_best, df_best_perf_final = encoded_data_model_run(
                obj_data=obj_data, 
                class_model=ModelRF, 
                percentage_test=0.2, 
                dict_perf={
                    "ACCURACY": accuracy, 
                    "AUC_MACRO": auc, 
                    "AUC_WEIGHTED": aucW, 
                    "F1_MACRO": f1, 
                    "F1_WEIGHTED": f1W
                },
                cv_folds=5, 
                verbose=True
            )

            df_performance.to_csv("../output/RF/encoded_" + DATASET_NAME + ".csv", sep=",", index=False)
            df_perf_best.to_csv("../output/RF/encoded_" + DATASET_NAME + "_best.csv", sep=",", index=False)
            df_best_perf_final.to_csv("../output/RF/encoded_" + DATASET_NAME + "_best_final.csv", sep=",", index=False)
    
    # # RF on original data
    # for DATASET_NAME in DATASET_NAMES:
    #     obj_data = Data(name=DATASET_NAME, generator=GENERATOR_NAME)
    #     obj_data.load()

    #     print("Original data", DATASET_NAME)
    #     df_performance, _, df_best_perf_final = data_model_run(
    #         obj_data=obj_data, 
    #         class_model=ModelRF, 
    #         percentage_test=0.2, 
    #         dict_perf={
    #             "ACCURACY": accuracy, 
    #             "AUC_MACRO": auc, 
    #             "AUC_WEIGHTED": aucW, 
    #             "F1_MACRO": f1, 
    #             "F1_WEIGHTED": f1W
    #         },
    #         verbose=True
    #     )

    #     df_performance.to_csv("../output/RF/" + DATASET_NAME + ".csv", sep=",", index=False)
    #     df_perf_best.to_csv("../output/RF/" + DATASET_NAME + "_best.csv", sep=",", index=False)
    #     df_best_perf_final.to_csv("../output/RF/" + DATASET_NAME + "_best_final.csv", sep=",", index=False)


