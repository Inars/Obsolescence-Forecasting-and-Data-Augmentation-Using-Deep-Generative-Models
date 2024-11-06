import os
import pandas as pd
import plotly.graph_objects as go

from sdv.metadata import SingleTableMetadata
from sdmetrics.single_table import BinaryAdaBoostClassifier, BinaryDecisionTreeClassifier, BinaryLogisticRegression, BinaryMLPClassifier    

def main():
    # DATASET_NAMES = ['moons','arrow','phone']
    DATASET_NAMES = ['arrow','phone']

    ctgan_babc = []
    tvae_babc = []
    nf_babc = []
    ctgan_bdtc = []
    tvae_bdtc = []
    nf_bdtc = []
    ctgan_blr = []
    tvae_blr = []
    nf_blr = []
    ctgan_bmlpc = []
    tvae_bmlpc = []
    nf_bmlpc = []
    for DATASET_NAME in DATASET_NAMES:
        print("Dataset:", DATASET_NAME)
        FOLDER_NAME = '../data/'
        if DATASET_NAME == 'moons':
            FOLDER_NAME += 'moons.csv'
        else:
            FOLDER_NAME += 'encoded/' + DATASET_NAME + '.csv'

        # load generated data
        ctgan_data = pd.read_csv('../data/discriminated/CTGAN_' + DATASET_NAME + '.csv')
        tvae_data = pd.read_csv('../data/discriminated/TVAE_' + DATASET_NAME + '.csv')
        nf_data = pd.read_csv('../data/discriminated/NF_' + DATASET_NAME + '.csv')

        # load real data
        real_data = pd.read_csv(FOLDER_NAME)

        # load metadata
        metadata = SingleTableMetadata.load_from_json(filepath='../data/metadata/discriminated/' + DATASET_NAME + '.json')
        metadata = metadata.to_dict()

        # Computing BinaryAdaBoostClassifier
        f1 = BinaryAdaBoostClassifier.compute(
            test_data=real_data,
            train_data=ctgan_data,
            target='label',
            metadata=metadata
        )
        ctgan_babc.append(f1)
        f1 = BinaryAdaBoostClassifier.compute(
            test_data=real_data,
            train_data=tvae_data,
            target='label',
            metadata=metadata
        )
        tvae_babc.append(f1)
        f1 = BinaryAdaBoostClassifier.compute(
            test_data=real_data,
            train_data=nf_data,
            target='label',
            metadata=metadata
        )
        nf_babc.append(f1)

        # Computing BinaryDecisionTreeClassifier
        f1 = BinaryDecisionTreeClassifier.compute(
            test_data=real_data,
            train_data=ctgan_data,
            target='label',
            metadata=metadata
        )
        ctgan_bdtc.append(f1)
        f1 = BinaryDecisionTreeClassifier.compute(
            test_data=real_data,
            train_data=tvae_data,
            target='label',
            metadata=metadata
        )
        tvae_bdtc.append(f1)
        f1 = BinaryDecisionTreeClassifier.compute(
            test_data=real_data,
            train_data=nf_data,
            target='label',
            metadata=metadata
        )
        nf_bdtc.append(f1)

        # Computing BinaryLogisticRegression
        f1 = BinaryLogisticRegression.compute(
            test_data=real_data,
            train_data=ctgan_data,
            target='label',
            metadata=metadata
        )
        ctgan_blr.append(f1)
        f1 = BinaryLogisticRegression.compute(
            test_data=real_data,
            train_data=tvae_data,
            target='label',
            metadata=metadata
        )
        tvae_blr.append(f1)
        f1 = BinaryLogisticRegression.compute(
            test_data=real_data,
            train_data=nf_data,
            target='label',
            metadata=metadata
        )
        nf_blr.append(f1)

        # Computing BinaryMLPClassifier
        f1 = BinaryMLPClassifier.compute(
            test_data=real_data,
            train_data=ctgan_data,
            target='label',
            metadata=metadata
        )
        ctgan_bmlpc.append(f1)
        f1 = BinaryMLPClassifier.compute(
            test_data=real_data,
            train_data=tvae_data,
            target='label',
            metadata=metadata
        )
        tvae_bmlpc.append(f1)
        f1 = BinaryMLPClassifier.compute(
            test_data=real_data,
            train_data=nf_data,
            target='label',
            metadata=metadata
        )
        nf_bmlpc.append(f1)

        
    # Plot ML Efficiency
    categories = ['Adaptive Boosting','Decision Tree','Logistic Regression','Multilayer Perceptron', 'Adaptive Boosting']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[ctgan_babc[0], ctgan_bdtc[0], ctgan_blr[0], ctgan_bmlpc[0], ctgan_babc[0]],
        theta=categories,
        name='CTGAN'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[tvae_babc[0], tvae_bdtc[0], tvae_blr[0], tvae_bmlpc[0], tvae_babc[0]],
        theta=categories,
        name='TVAE'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[nf_babc[0], nf_bdtc[0], nf_blr[0], nf_bmlpc[0], nf_babc[0]],
        theta=categories,
        # fill='toself',
        name='Real NVP'
    ))
    fig.update_layout(
        title="ML Efficiency (Arrow)",
        font=dict(family="Times New Roman"),
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0.79, 0.91]
            )),
        showlegend=True
    )
    fig.show()
    fig.write_image("../media/ml_efficiency_arrow.png")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[ctgan_babc[1], ctgan_bdtc[1], ctgan_blr[1], ctgan_bmlpc[1], ctgan_babc[1]],
        theta=categories,
        name='CTGAN'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[tvae_babc[1], tvae_bdtc[1], tvae_blr[1], tvae_bmlpc[1], tvae_babc[1]],
        theta=categories,
        name='TVAE'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[nf_babc[1], nf_bdtc[1], nf_blr[1], nf_bmlpc[1], nf_babc[1]],
        theta=categories,
        # fill='toself',
        name='Real NVP'
    ))
    fig.update_layout(
        title="ML Efficiency (Phone)",
        font=dict(family="Times New Roman"),
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0.91, 0.98]
            )),
        showlegend=True
    )
    fig.show()
    fig.write_image("../media/ml_efficiency_phone.png")

    print("CTGAN F1 Score on Arrow: AdaBoost Classifier " + str(ctgan_babc[0]) + ", Decision Tree Classifier " + str(ctgan_bdtc[0]) + ", Logistic Regression " + str(ctgan_blr[0]) + ", MLP Classifier " + str(ctgan_bmlpc[0]))
    print("TVAE F1 Score on Arrow: AdaBoost Classifier " + str(tvae_babc[0]) + ", Decision Tree Classifier " + str(tvae_bdtc[0]) + ", Logistic Regression " + str(tvae_blr[0]) + ", MLP Classifier " + str(tvae_bmlpc[0]))
    print("NF F1 Score on Arrow: AdaBoost Classifier " + str(nf_babc[0]) + ", Decision Tree Classifier " + str(nf_bdtc[0]) + ", Logistic Regression " + str(nf_blr[0]) + ", MLP Classifier " + str(nf_bmlpc[0]))
    print("CTGAN F1 Score on Phone: AdaBoost Classifier " + str(ctgan_babc[1]) + ", Decision Tree Classifier " + str(ctgan_bdtc[1]) + ", Logistic Regression " + str(ctgan_blr[1]) + ", MLP Classifier " + str(ctgan_bmlpc[1]))
    print("TVAE F1 Score on Phone: AdaBoost Classifier " + str(tvae_babc[1]) + ", Decision Tree Classifier " + str(tvae_bdtc[1]) + ", Logistic Regression " + str(tvae_blr[1]) + ", MLP Classifier " + str(tvae_bmlpc[1]))
    print("NF F1 Score on Phone: AdaBoost Classifier " + str(nf_babc[1]) + ", Decision Tree Classifier " + str(nf_bdtc[1]) + ", Logistic Regression " + str(nf_blr[1]) + ", MLP Classifier " + str(nf_bmlpc[1]))
    print("Done!")
        

if __name__ == '__main__':
    main()