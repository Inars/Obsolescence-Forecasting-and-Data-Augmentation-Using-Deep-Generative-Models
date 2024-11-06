import pandas as pd
import plotly.graph_objects as go
import numpy as np

from sdv.metadata import SingleTableMetadata
from sdmetrics.visualization import get_column_plot
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.single_column import KSComplement, RangeCoverage
from sdmetrics.single_table import LogisticDetection, SVCDetection, GMLogLikelihood
from scipy.stats import wasserstein_distance_nd, ksone
from scipy import stats

def ks_critical_value(n_trials, alpha):
    return ksone.ppf(1-alpha/2, n_trials)

def print_test(value, DATASET_NAME, MODEL_NAME, TEST_NAME):
    print(MODEL_NAME + ": " + TEST_NAME + " on " + DATASET_NAME + " --- " + str(value))

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
 
    # x-data for the ECDF: x
    x = np.sort(data)
 
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
 
    return x, y

def main():
    # DATASET_NAMES = ['moons','arrow','phone']
    DATASET_NAMES = ['arrow','phone']

    for DATASET_NAME in DATASET_NAMES:
        print("Dataset:", DATASET_NAME)
        FOLDER_NAME = '../data/'
        if DATASET_NAME == 'moons':
            FOLDER_NAME += 'moons.csv'
        else:
            FOLDER_NAME += 'encoded/' + DATASET_NAME + '.csv'

        # load generated data
        ctgan_data = pd.read_csv('../data/generated/CTGAN/' + DATASET_NAME + '.csv')
        tvae_data = pd.read_csv('../data/generated/TVAE/' + DATASET_NAME + '.csv')
        nf_data = pd.read_csv('../data/generated/NF/' + DATASET_NAME + '.csv')

        # load real data
        real_data = pd.read_csv(FOLDER_NAME)
        real_data = real_data.drop('label', axis=1)

        # load metadata
        metadata = SingleTableMetadata.load_from_json(filepath='../data/metadata/generated/' + DATASET_NAME + '.json')
        metadata = metadata.to_dict()

        # sample data
        real_data_sample = real_data.sample(n=500, random_state=1)
        ctgan_data_sample = ctgan_data.sample(n=500, random_state=1)
        tvae_data_sample = tvae_data.sample(n=500, random_state=1)
        nf_data_sample = nf_data.sample(n=500, random_state=1)

        # load models
        # ctgan_synthesizer = CTGANSynthesizer.load(filepath='../models/generative/ctgan_moons.pkl')
        # tvae_synthesizer = TVAESynthesizer.load(filepath='../models/generative/tvae_moons.pkl')

        # Kolmogorov-Smirnov test
        ks1 = stats.ks_2samp(real_data['x1'], ctgan_data['x1'], alternative='two-sided')
        ks2 = stats.ks_2samp(real_data['x2'], ctgan_data['x2'], alternative='two-sided')
        print_test((ks1.statistic + ks2.statistic) / 2, DATASET_NAME, "CTGAN", "KS Test Difference Statistic")
        print_test((ks1.pvalue + ks2.pvalue) / 2, DATASET_NAME, "CTGAN", "KS Test p Value")
        print_test(ks_critical_value(real_data['x1'].shape[0], 0.05), DATASET_NAME, "CTGAN", "KS Test Critical Value")
        ks1 = stats.ks_2samp(real_data['x1'], tvae_data['x1'], alternative='two-sided')
        ks2 = stats.ks_2samp(real_data['x2'], tvae_data['x2'], alternative='two-sided')
        print_test((ks1.statistic + ks2.statistic) / 2, DATASET_NAME, "TVAE", "KS Test Difference Statistic")
        print_test((ks1.pvalue + ks2.pvalue) / 2, DATASET_NAME, "TVAE", "KS Test p Value")
        print_test(ks_critical_value(real_data['x1'].shape[0], 0.05), DATASET_NAME, "TVAE", "KS Test Critical Value")
        ks1 = stats.ks_2samp(real_data['x1'], nf_data['x1'], alternative='two-sided')
        ks2 = stats.ks_2samp(real_data['x2'], nf_data['x2'], alternative='two-sided')
        print_test((ks1.statistic + ks2.statistic) / 2, DATASET_NAME, "NF", "KS Test Difference Statistic")
        print_test((ks1.pvalue + ks2.pvalue) / 2, DATASET_NAME, "NF", "KS Test p Value")
        print_test(ks_critical_value(real_data['x1'].shape[0], 0.05), DATASET_NAME, "NF", "KS Test Critical Value")
        print("Kolonogorov-Smirnov Test Done")

        # Wasserstein Distance
        wd = wasserstein_distance_nd(real_data_sample, ctgan_data_sample)
        print_test(wd, DATASET_NAME, "CTGAN", "Wasserstein Distance")
        wd = wasserstein_distance_nd(real_data_sample, tvae_data_sample)
        print_test(wd, DATASET_NAME, "TVAE", "Wasserstein Distance")
        wd = wasserstein_distance_nd(real_data_sample, nf_data_sample)
        print_test(wd, DATASET_NAME, "NF", "Wasserstein Distance")
        print("Wasserstein Distance Done")

        # Correlation Similarity Score
        pearson = 1 - CorrelationSimilarity.compute(
            real_data=real_data,
            synthetic_data=ctgan_data,
            coefficient='Pearson'
        )
        print_test(pearson, DATASET_NAME, "CTGAN", "Pearson Correlation Similarity")
        pearson = 1 - CorrelationSimilarity.compute(
            real_data=real_data,
            synthetic_data=tvae_data,
            coefficient='Pearson'
        )
        print_test(pearson, DATASET_NAME, "TVAE", "Pearson Correlation Similarity")
        pearson = 1 - CorrelationSimilarity.compute(
            real_data=real_data,
            synthetic_data=nf_data,
            coefficient='Pearson'
        )
        print_test(pearson, DATASET_NAME, "NF", "Pearson Correlation Similarity")
        print("Correlation Similarity Done")

        # Range Coverage
        range1 = RangeCoverage.compute(
            real_data=real_data["x1"],
            synthetic_data=ctgan_data["x1"]
        )
        range2 = RangeCoverage.compute(
            real_data=real_data["x2"],
            synthetic_data=ctgan_data["x2"]
        )
        print_test((range1 + range2) / 2, DATASET_NAME, "CTGAN", "Range Coverage")
        range1 = RangeCoverage.compute(
            real_data=real_data["x1"],
            synthetic_data=tvae_data["x1"]
        )
        range2 = RangeCoverage.compute( 
            real_data=real_data["x2"],
            synthetic_data=tvae_data["x2"]
        )
        print_test((range1 + range2) / 2, DATASET_NAME, "TVAE", "Range Coverage")
        range1 = RangeCoverage.compute(
            real_data=real_data["x1"],
            synthetic_data=nf_data["x1"]
        )
        range2 = RangeCoverage.compute(
            real_data=real_data["x2"],
            synthetic_data=nf_data["x2"]
        )
        print_test((range1 + range2) / 2, DATASET_NAME, "NF", "Range Coverage")
        print("Range Coverage Done")

        # Gaussian Mixture Likelihood
        likelihood = GMLogLikelihood.compute(
            real_data=real_data,
            synthetic_data=ctgan_data,
        )
        print_test(likelihood, DATASET_NAME, "CTGAN", "Gaussian Mixture Likelihood")
        likelihood = GMLogLikelihood.compute(
            real_data=real_data,
            synthetic_data=tvae_data,
        )
        print_test(likelihood, DATASET_NAME, "TVAE", "Gaussian Mixture Likelihood")
        likelihood = GMLogLikelihood.compute(
            real_data=real_data,
            synthetic_data=nf_data,
        )
        print_test(likelihood, DATASET_NAME, "NF", "Gaussian Mixture Likelihood")
        print("Gaussian Mixture Likelihood Done")

        # LogisticRegression Detection
        ld = LogisticDetection.compute(
            real_data=real_data,
            synthetic_data=ctgan_data,
            metadata=metadata
        )
        print_test(ld, DATASET_NAME, "CTGAN", "LogisticRegression Detection")
        ld = LogisticDetection.compute(
            real_data=real_data,
            synthetic_data=tvae_data,
            metadata=metadata
        )
        print_test(ld, DATASET_NAME, "TVAE", "LogisticRegression Detection")
        ld = LogisticDetection.compute(
            real_data=real_data,
            synthetic_data=nf_data,
            metadata=metadata
        )
        print_test(ld, DATASET_NAME, "NF", "LogisticRegression Detection")
        print("LogisticRegression Detection Done")

        # SupportVectorClassifier Detection
        svc = SVCDetection.compute(
            real_data=real_data,
            synthetic_data=ctgan_data,
            metadata=metadata
        )
        print_test(svc, DATASET_NAME, "CTGAN", "SVC Detection")
        svc = SVCDetection.compute(
            real_data=real_data,
            synthetic_data=tvae_data,
            metadata=metadata
        )
        print_test(svc, DATASET_NAME, "TVAE", "SVC Detection")
        svc = SVCDetection.compute(
            real_data=real_data,
            synthetic_data=nf_data,
            metadata=metadata
        )
        print_test(svc, DATASET_NAME, "NF", "SVC Detection")
        print("SVC Detection Done")

        # Data Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=real_data_sample["x1"], y=real_data_sample["x2"],
            name='Real',
            mode='markers',
            marker_color='rgb(0, 0, 54)'
        ))
        fig.add_trace(go.Scatter(
            x=ctgan_data_sample["x1"], y=ctgan_data_sample["x2"],
            name='Synthetic',
            mode='markers',
            marker_color='rgb(1, 224, 201)'
        ))
        fig.update_layout(
            title=None,
            xaxis_title="x1",
            yaxis_title="x2",
            font=dict(family="Times New Roman")
        )
        # fig.show()
        fig.write_image("../media/ctgan_" + DATASET_NAME + "_distribution.png")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=real_data_sample["x1"], y=real_data_sample["x2"],
            name='Real',
            mode='markers',
            marker_color='rgb(0, 0, 54)'
        ))
        fig.add_trace(go.Scatter(
            x=tvae_data_sample["x1"], y=tvae_data_sample["x2"],
            name='Synthetic',
            mode='markers',
            marker_color='rgb(1, 224, 201)'
        ))
        fig.update_layout(
            title=None,
            xaxis_title="x1",
            yaxis_title="x2",
            font=dict(family="Times New Roman")
        )
        # fig.show()
        fig.write_image("../media/tvae_" + DATASET_NAME + "_distribution.png")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=real_data_sample["x1"], y=real_data_sample["x2"],
            name='Real',
            mode='markers',
            marker_color='rgb(0, 0, 54)'
        ))
        fig.add_trace(go.Scatter(
            x=nf_data_sample["x1"], y=nf_data_sample["x2"],
            name='Synthetic',
            mode='markers',
            marker_color='rgb(1, 224, 201)'
        ))
        fig.update_layout(
            title=None,
            xaxis_title="x1",
            yaxis_title="x2",
            font=dict(family="Times New Roman")
        )
        # fig.show()
        fig.write_image("../media/nf_" + DATASET_NAME + "_distribution.png")

        # fig = get_column_plot(
        #     real_data=real_data,
        #     synthetic_data=ctgan_data,
        #     column_name='x1',
        #     plot_type='distplot'
        # )
        # fig.update_layout(
        #     title="CTGAN: Real vs. Synthetic Data for column x1 (" + DATASET_NAME + ")",
        #     xaxis_title="Value",
        #     yaxis_title="Density",
        #     font=dict(
        #         family="Times New Roman",
        #         size=12
        #     )
        # )
        # # fig.show()
        # fig.write_image("../media/CTGAN/" + DATASET_NAME + "_distribution_x1.png")

        print("Figures Generation Done")
    
    print("Generating Plots")

    # # Plot Kolmogorov-Smirnov test
    # fig = go.Figure()
    # fig.add_trace(go.Histogram(x=ctgan_ks_X, y=ctgan_ks_Y, name="CTGAN", histfunc='sum'))
    # fig.add_trace(go.Histogram(x=tvae_ks_X, y=tvae_ks_Y, name="TVAE", histfunc='sum'))
    # fig.add_trace(go.Histogram(x=nf_ks_X, y=nf_ks_Y, name="NF", histfunc='sum'))
    # fig.update_layout(
    #     title="Kolmogorov-Smirnov Test",
    #     xaxis_title="Dataset",
    #     yaxis_title="Scores",
    #     font=dict(family="Times New Roman")
    # )
    # fig.show()
    # fig.write_image("../media/ks.png")

    print("All Done")


if __name__ == "__main__":
    main()
    