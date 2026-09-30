# Source-disjoint evaluation results

## Dataset and split verification

The copied manifest contains 17,056 cleaned multimodal samples. Each subreddit belongs to one partition only and both labels appear in each split. All image paths resolve, every image opens correctly and the regional-unavailable placeholder is absent. Sample IDs are unique and all text fields are non-empty.

| split | manifest_samples | retained_samples | removed_samples | removed_class_0 | removed_class_1 | class_0 | class_1 | class_1_rate | sources |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 10524 | 10524 | 0 | 0 | 0 | 4955 | 5569 | 0.5292 | 17 |
| validation | 3793 | 3793 | 0 | 0 | 0 | 1379 | 2414 | 0.6364 | 2 |
| test | 2739 | 2739 | 0 | 0 | 0 | 1241 | 1498 | 0.5469 | 3 |

The source-disjoint manifest was built from the existing cleaned multimodal pool. Its manifest count is also its final usable count, so this run removed no further samples from either class.

## Training and locked validation choices

RoBERTa-base and ResNet-50 were trained from their pretrained base weights using only the source-disjoint training split. Subreddit, domain, timestamp and engagement fields were not model inputs. Both models used a maximum of three epochs and the best checkpoint was chosen by validation macro F1.

RoBERTa checkpoint: `F:\data sci\project\v2\experiments\source_disjoint\checkpoints\roberta`. Epoch 2 was selected with validation macro F1 0.4101.

ResNet-50 checkpoint: `F:\data sci\project\v2\experiments\source_disjoint\checkpoints\resnet50\best_model.pt`. Epoch 1 was selected with validation macro F1 0.6158.

The validation search selected α = 0.00 for RoBERTa and 1.00 for ResNet-50 at threshold 0.50. There was no tie. Since α is zero, the locked late-fusion model is numerically identical to ResNet-50. It should not be described as a benefit from combining modalities.

The full 21-point search is saved in `metrics/validation_alpha_search.csv`.

## Source-disjoint test performance

| model | accuracy | precision | recall | f1 | macro_f1 | recall_class_0 | recall_class_1 | roc_auc | pr_auc | brier | log_loss | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| roberta | 0.7601 | 0.8081 | 0.7363 | 0.7705 | 0.7596 | 0.7889 | 0.7363 | 0.8401 | 0.8576 | 0.1825 | 0.5806 | 0.1189 |
| resnet50 | 0.6842 | 0.6922 | 0.7610 | 0.7250 | 0.6771 | 0.5915 | 0.7610 | 0.7453 | 0.7502 | 0.2040 | 0.6089 | 0.0572 |
| late_fusion | 0.6842 | 0.6922 | 0.7610 | 0.7250 | 0.6771 | 0.5915 | 0.7610 | 0.7453 | 0.7502 | 0.2040 | 0.6089 | 0.0572 |

RoBERTa remained the stronger unimodal model on the three unseen test subreddits. ResNet-50 retained predictive information but scored below RoBERTa. The validation-selected late fusion also scored below RoBERTa because validation assigned zero weight to text.

## Calibration

| model | brier | log_loss | ece |
| --- | --- | --- | --- |
| roberta | 0.1825 | 0.5806 | 0.1189 |
| resnet50 | 0.2040 | 0.6089 | 0.0572 |
| late_fusion | 0.2040 | 0.6089 | 0.0572 |

ResNet-50 and the locked fusion had lower ECE, while RoBERTa had lower Brier score and log loss. ECE uses the project convention of ten confidence bins from 0.5 to 1.0.

## McNemar test

Both RoBERTa and fusion were correct on 1,458 cases and both were wrong on 241. RoBERTa alone was correct on 624; fusion alone was correct on 416. The exact two-sided McNemar p-value was 1.1965e-10.

The direction favours RoBERTa because it won more discordant cases.

## Paired bootstrap confidence intervals

| metric | difference | ci_lower | ci_upper | valid_iterations | requested_iterations |
| --- | --- | --- | --- | --- | --- |
| accuracy | -0.0759 | -0.0989 | -0.0533 | 5000 | 5000 |
| f1 | -0.0456 | -0.0670 | -0.0240 | 5000 | 5000 |
| macro_f1 | -0.0825 | -0.1054 | -0.0593 | 5000 | 5000 |
| roc_auc | -0.0948 | -0.1165 | -0.0727 | 5000 | 5000 |
| pr_auc | -0.1073 | -0.1315 | -0.0827 | 5000 | 5000 |

The fusion-minus-RoBERTa macro F1 difference was -0.0825, with a 95% paired bootstrap interval of [-0.1054, -0.0593]. The interval does not cross zero and favours RoBERTa.

## Error complementarity

RoBERTa and ResNet-50 disagreed on 1,040 test cases. RoBERTa alone was correct on 624; ResNet-50 alone was correct on 416. The locked fusion recovered all 416 image-correct/text-wrong cases because it equals the image model, but it preserved none of the 624 text-correct/image-wrong cases.

The disagreement count shows that the modalities make different errors. The selected coefficient did not use that complementarity on test because the two validation subreddits favoured an image-only decision.

## Change from the ordinary locked holdout

| model | metric | ordinary_locked_holdout | source_disjoint | change_source_disjoint_minus_ordinary |
| --- | --- | --- | --- | --- |
| roberta | accuracy | 0.8046 | 0.7601 | -0.0445 |
| roberta | macro_f1 | 0.8000 | 0.7596 | -0.0404 |
| roberta | roc_auc | 0.8846 | 0.8401 | -0.0445 |
| roberta | pr_auc | 0.9027 | 0.8576 | -0.0451 |
| resnet50 | accuracy | 0.7382 | 0.6842 | -0.0540 |
| resnet50 | macro_f1 | 0.7275 | 0.6771 | -0.0504 |
| resnet50 | roc_auc | 0.7976 | 0.7453 | -0.0523 |
| resnet50 | pr_auc | 0.8053 | 0.7502 | -0.0551 |
| late_fusion | accuracy | 0.8242 | 0.6842 | -0.1400 |
| late_fusion | macro_f1 | 0.8188 | 0.6771 | -0.1417 |
| late_fusion | roc_auc | 0.8993 | 0.7453 | -0.1540 |
| late_fusion | pr_auc | 0.9097 | 0.7502 | -0.1595 |

RoBERTa macro F1 fell from 0.8000 to 0.7596, a change of -0.0404. ResNet-50 fell from 0.7275 to 0.6771, a change of -0.0504. Locked late fusion fell from 0.8188 to 0.6771, a change of -0.1417.

## Interpretation

Removing source overlap produced a clear generalisation penalty for all three reported systems. RoBERTa still transferred better than ResNet-50 on the test sources. The image model retained useful signal, shown by its ROC AUC and the cases it alone classified correctly.

Late fusion did not outperform RoBERTa in this experiment. Its validation-selected coefficient removed the text contribution, then underperformed RoBERTa on test by 0.0825 macro F1. The confidence interval and McNemar result support that direction. This also exposes a limit in the split: validation contains only two subreddits and test contains three. Model and fusion selection can depend heavily on which whole sources enter each partition. The result is still useful because it shows how performance and modality weighting change when source overlap is removed.
