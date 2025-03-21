import argparse 
import json
import numpy as np 
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Tuple

from collections import Counter
import numpy as np 
import scipy 
# from calibration_metric.metric import ECEMetric
from scipy import stats

class Metric:
    """
    Abstract class for metrics. 
    """
    def __init__(self, 
                name: str,
                n_bins: int = 20,
                weighted: bool = False,
                weight_key: str = "normalized_count",
                binning_strategy: str = "uniform",
        ):
        if weighted:
            name = f"Weighted {name}"
        self.name = name
        self.n_bins = n_bins
        self.weighted = weighted
        self.weight_key = weight_key
        self.binning_strategy = binning_strategy

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        raise NotImplementedError

    def weight_by_count(self, p_correct: np.array, p_model: np.array, normalized_counts: np.array) -> np.array:
        """
        Weight the metric by the number of examples in each bin.

        Parameters
        ----------
        p_correct : np.array
            Array of the average number of correct examples in each bin.
        p_model : np.array
            Array of the average confidence in each bin.
        normalized_counts : np.array
            Array of the number of examples in each bin, normalized by the total number of examples.
    
        Returns
        -------
        weighted_metric : np.array
            Weighted absolute error between p_correct and p_model.
        """
        abs_error = np.abs(p_correct - p_model) 
        weighted_mean_error = np.sum(abs_error * normalized_counts)
        return weighted_mean_error

    def uniform_bin(self,
                    top_probs: np.array,
                    is_correct: np.array):
        """Implements naive uniform binning strategy"""
        (values, 
        bins, 
        bin_number) = stats.binned_statistic(
            top_probs, 
            is_correct, 
            statistic='mean', 
            bins=self.n_bins
        )
        bin_number = bin_number - 1
        return values, bins, bin_number

    def adaptive_bin(self,
                    top_probs: np.array,
                    is_correct: np.array):
        """Implements adaptive binning strategy as in https://github.com/yding5/AdaptiveBinning
        (adapted from https://github.com/yding5/AdaptiveBinning/blob/master/AdaptiveBinning.py)"""
        # zip and sort 
        zipped = sorted(list(zip(top_probs, is_correct)), key=lambda x: x[0], reverse=True)
        n_total = len(zipped)

        z = 1.645
        num = [0 for i in range(n_total)]
        final_num = [0 for i in range(n_total)]
        correct = [0 for i in range(n_total)]
        confidence = [0 for i in range(n_total)]
        conf_min = [1 for i in range(n_total)]
        conf_max = [0 for i in range(n_total)]
        accuracy = [0 for i in range(n_total)]

        ind = 0
        target_num_samples = float("inf")

        # traverse all samples for initial binning
        for i, (prob, is_correct) in enumerate(zipped):
            assert(prob <= 1.0 and prob >= 0.0)
            # merge the last bin if too small
            if num[ind] > target_num_samples and \
                (n_total - i) > 40 \
                and conf_min[ind] - zipped[-1][0] > 0.05:
                ind += 1
                target_num_samples = float("inf")

            num[ind] += 1
            confidence[ind] += prob
            
            if is_correct:
                correct[ind] += 1
            
            conf_min[ind] = min(conf_min[ind], prob)
            conf_max[ind] = max(conf_max[ind], prob)

            # get target number of samples in the bin
            if conf_max[ind] == conf_min[ind]:
                target_num_samples = float("inf")
            else:
                target_num_samples = (z / (conf_max[ind]-conf_min[ind])) ** 2 * 0.25
        # pdb.set_trace()
        n_bins = ind + 1
        # get final binning
        if target_num_samples - num[ind] > 0:
            needed = target_num_samples - num[ind]
            extract = [0 for i in range(n_bins - 1)]
            final_num[n_bins - 1] = num[n_bins - 1]
            for i in range(n_bins - 1):
                extract[i] = int(needed * num[ind] / n_total)
                final_num[i] = num[i] - extract[i]
                final_num[n_bins - 1] += extract[i]
        else:
            final_num = num
        final_num = final_num[:n_bins]

        # re-intialize
        num = [0 for i in range(n_bins)]
        correct = [0 for i in range(n_bins)]
        confidence = [0 for i in range(n_bins)]
        conf_min = [1 for i in range(n_bins)]
        conf_max = [0 for i in range(n_bins)]
        accuracy = [0 for i in range(n_bins)]
        gap = [0 for i in range(n_bins)]
        neg_gap = [0 for i in range(n_bins)]

        ind = 0
        bin_number = []
        for i, (prob, is_correct) in enumerate(zipped):
            bin_number.append(ind)
            num[ind] += 1
            confidence[ind] += prob

            if is_correct: 
                correct[ind] += 1
            conf_min[ind] = min(conf_min[ind], prob)
            conf_max[ind] = max(conf_max[ind], prob)

            if num[ind] == final_num[ind]:
                confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
                accuracy[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0

                if confidence[ind] - accuracy[ind] > 0:
                    gap[ind] = confidence[ind] - accuracy[ind]
                else:
                    neg_gap[ind] = confidence[ind] - accuracy[ind]
                ind += 1

        values = np.array(accuracy)
        bin_edges = np.array(confidence)
        bin_number = np.array(bin_number)
        return values, bin_edges, bin_number

    def bin_preds(self, 
                 top_probs: np.array, 
                 is_correct: np.array): 
        """
        Bin predicted probabilities and correct binary labels into n_bins.
        Binning is done by predited probability, and each bin's value is 
        the average number of correct examples in that bin.

        Parameters
        ----------
        top_probs : np.array
            Array of predicted probabilities for each timestep across all examples.
        is_correct : np.array
            Array of whether each timestep is correct for each timestep across all examples.

        Returns
        -------
        values : np.array
            (n_bins, 1), Array of the average number of correct examples in each bin.
        bin_edges : np.array
            (n_bins, 1), Array of the bin edges (probabilities)
        bin_number : np.array
            (n_examples, 1), Array of the bin number for each example.
        """ 
        try:
            assert(top_probs.shape[0] == is_correct.shape[0])
        except AssertionError:
            raise AssertionError(f"top_probs and is_correct must have the same length, got {top_probs.shape} and {is_correct.shape} respectively.")

        # bin predicted probs in n_bins bins 
        # (values, 
        # bins, 
        # bin_number) = stats.binned_statistic(
        #     top_probs, 
        #     is_correct, 
        #     statistic='mean', 
        #     bins=self.n_bins
        # )
        if self.binning_strategy == "uniform":
            values, bins, bin_number = self.uniform_bin(top_probs, is_correct)
        elif self.binning_strategy == "adaptive": 
            values, bins, bin_number = self.adaptive_bin(top_probs, is_correct)
        else:
            raise ValueError(f"Invalid binning strategy: {self.binning_strategy}")

        if any(np.isnan(values)):
            check_size_warning(top_probs, is_correct, self.name)
            logger.warn(f"NaN values in values from insufficient samples. Try decreasing n_bins or increasing the number of samples.")
            pre_values = np.array([x for x in values])
            pre_bins = np.array([x for x in bins])
            value_idxs = [i for i in range(len(values)) if not np.isnan(values[i])]
            nan_idxs = [i for i in range(len(values)) if np.isnan(values[i])]
            values = values[value_idxs]
            bins = bins[value_idxs]
            logger.warn(f"Reducing number of bins to {len(values)} by dropping NaN values at {nan_idxs} for bins {pre_bins[nan_idxs]}.")


        return (values, bins, bin_number)

    def bins_to_df(self, 
        values: np.array,
        bin_edges: np.array,
        bin_number: np.array,
        ) -> pd.DataFrame:
        """
        Convert the output of bin_preds to a pandas dataframe.
        DataFrame has following columns:
        - prob_model: the probability for the bin
        - prob_correct: the average number of correct examples in the bin
        - count: the number of examples in the bin
        """
        # create LUT for bin number to number of items in that bin 
        bin_lookup = Counter(bin_number)
        # instantiate df 
        # df = pd.DataFrame(columns=["prob_model", "prob_correct", "count"])
        # populate df
        df_data = []
        for i, (val, edge_start, bin_num) in enumerate(zip(values, bin_edges, bin_number)):
            if self.binning_strategy == "uniform":
                edge_end = bin_edges[i+1]
                midpoint = (edge_start + edge_end) / 2
            else:
                midpoint = edge_start

            df_data.append({"prob_model": midpoint, 
                            "prob_correct": val, 
                            "count": bin_lookup[i]})
        df = pd.DataFrame.from_dict(df_data)
        df['normalized_count'] = df['count'] / df['count'].sum()
        df['log_count'] = np.log(df['count']) 
        # NOTE: this is not the same as the log of the normalized count; it is intended to
        # discount high count bins.
        df['normalized_log_count'] = df['log_count'] / df['log_count'].sum()
        return df




class ECEMetric(Metric):
    """
    Computes expected calibration error (https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf)
    """
    def __init__(self,
                n_bins: int = 20,
                weighted: bool = True,
                return_df: bool = False,
                weight_key: str = "normalized_count",
                binning_strategy: str = "uniform"):
        super().__init__("ECE", n_bins, weighted, weight_key, binning_strategy)
        self.return_df = return_df

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        """
        Parameters
        ----------
        top_preds : np.array
            Array of predicted probabilities for each timestep across all examples.
        is_correct : np.array
            Array of whether each timestep is correct for each timestep across all examples.
        
        Returns
        -------
        ece : float
            The expected calibration error 
        """
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        p_model = df["prob_model"].values
        p_correct = df["prob_correct"].values
        # check_size_warning(p_correct, p_model, self.name)
        if self.weighted:
            norm_counts = df[self.weight_key].values
            ece = self.weight_by_count(p_correct, p_model, norm_counts)
        else:
            ece = np.mean(np.abs(p_model - p_correct))

        if self.return_df:
            return ece, df
        return ece 

class MCEMetric(Metric):
    """
    Computes maximum calibration error (https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf)
    """
    def __init__(self,
                n_bins: int = 20,
                weighted: bool = False,
                weight_key: str = "normalized_count",
                binning_strategy: str = "uniform"):
        super().__init__("MCE", n_bins, weighted, weight_key, binning_strategy)

    def __call__(self, 
                top_preds: np.array,
                is_correct: np.array) -> float:
        """
        Parameters
        ----------
        top_preds : np.array
            Array of predicted probabilities for each timestep across all examples.
        is_correct : np.array
            Array of whether each timestep is correct for each timestep across all examples.
        
        Returns
        -------
        mce : float
            The max calibration error 
        """
        values, bin_edges, bin_number = self.bin_preds(top_preds, is_correct) 
        df = self.bins_to_df(values, bin_edges, bin_number)
        p_model = df["prob_model"].values
        p_correct = df["prob_correct"].values
        check_size_warning(p_correct, p_model, self.name)

        mce = np.max(np.abs(p_model - p_correct))
        return mce 


def read_file(path):
    with open(path) as f1:
        data = [json.loads(x) for x in f1.readlines()]

    return data

def main(args):
    data = read_file(args.eval_file)
    data_by_type = {"trained": {"tp":[], "fp":[], "fn":[]}, "reference": {"tp":[], "fp":[], "fn":[]}}

    scores_for_auroc = {"trained": {"probs": [], "corrects": [], "nonanswers": []}, "reference": {"probs": [], "corrects": [], "nonanswers": []}}


    for row in data:
        ref_prob = row['reference_prob']
        trained_prob = row['trained_prob']
        ref_correct = row['reference_correct']
        trained_correct = row['trained_correct']

        # don't include examples where either model gave no answer
        if row['reference_answer'] == "NONE": 
            scores_for_auroc['reference']['nonanswers'].append(True)
        else:
            scores_for_auroc['reference']['nonanswers'].append(False)
        if row['trained_answer'] == "NONE": 
            scores_for_auroc['trained']['nonanswers'].append(True)
        else:
            scores_for_auroc['trained']['nonanswers'].append(False)  
        if args.skip_none:
            if row['reference_answer'] == "NONE" or row['trained_answer'] == "NONE":
                continue

        scores_for_auroc["trained"]["probs"].append(trained_prob)
        scores_for_auroc["trained"]["corrects"].append(trained_correct)
        scores_for_auroc["reference"]["probs"].append(ref_prob)
        scores_for_auroc["reference"]["corrects"].append(ref_correct)

        ref_accept = ref_prob > args.threshold
        trained_accept = trained_prob > args.threshold

        data_by_type["trained"]["tp"].append(trained_accept and trained_correct)
        data_by_type["trained"]["fp"].append(trained_accept and not trained_correct)
        data_by_type["trained"]["fn"].append(not trained_accept and trained_correct)

        data_by_type["reference"]["tp"].append(ref_accept and ref_correct)
        data_by_type["reference"]["fp"].append(ref_accept and not ref_correct)
        data_by_type["reference"]["fn"].append(not ref_accept and ref_correct)

    data_to_write = {}
    for model_type, model_data in data_by_type.items():
        tp = np.sum(model_data["tp"])
        fp = np.sum(model_data["fp"])
        fn = np.sum(model_data["fn"])

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)

        acc = np.mean(scores_for_auroc[model_type]['corrects'])

        print(f"Model Type: {model_type}")
        print(f"Precision: {prec*100:.2f}")
        print(f"Recall: {rec*100:.2f}")
        print(f"F1: {f1*100:.2f}")
        print(f"Accuracy: {acc*100:.2f}")
        abstention_rate = np.mean(scores_for_auroc[model_type]['nonanswers'])
        print(f"Rate of nonanswers: {np.mean(scores_for_auroc[model_type]['nonanswers'])*100:.2f}")

        preds, corrects = scores_for_auroc[model_type]["probs"], scores_for_auroc[model_type]["corrects"]
        auroc = roc_auc_score(corrects, preds)
        print(f"AUROC: {auroc}")

        # get ECE 
        try:
            metric = ECEMetric(n_bins=9)
            ece = metric(np.array(preds), np.array(corrects))
            print(f"ECE: {ece*100:.2f}")
        except IndexError:
            ece = np.nan


        print()
        data_to_write[model_type] = {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "auroc": auroc, "ece": ece, "abstenion": abstention_rate}

    # reference accuracy on nonanswers
    trained_nonanswers = scores_for_auroc["trained"]["nonanswers"]
    reference_corrects = scores_for_auroc["reference"]["corrects"]
    corrects_when_answered = [x for x, y in zip(reference_corrects, trained_nonanswers) if not y]
    corrects_when_not_answered = [x for x, y in zip(reference_corrects, trained_nonanswers) if y]
    print(f"reference accuracy on nonanswers when trained model did not answer: {np.mean(corrects_when_not_answered)*100:.2f}")
    print(f"reference accuracy on nonanswers when trained model did answer: {np.mean(corrects_when_answered)*100:.2f}")

    out_path = Path(args.eval_file).parent
    skip_str = "skip" if args.skip_none else "noskip"
    with open(out_path / f"eval_data_{skip_str}.json", "w") as f1:
        json.dump(data_to_write, f1)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--skip_none", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args)