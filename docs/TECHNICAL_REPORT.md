# Technical Proposal - TrustGuard

Cookie Squad | NTU Datathon 2026, Track 2

## 1. Problem Understanding

TrustGuard must rank rare fraudulent transactions while limiting disruption to legitimate customers. Training contains 20,000 transactions and 353 frauds (1.765%); the public test contains 12,000 unlabeled rows. Predicting every row as legitimate gives 98.235% training accuracy, so we prioritised precision-recall ranking and examined the workload/recall trade-off rather than accuracy.

## 2. Data Processing & Feature Engineering

We excluded IDs, checked duplicates and missingness, normalised categories, imputed missing values and clipped numeric inputs. Mean feature missingness is 0.52% in train versus 1.81% in test. Features describe log amounts and account age, amount relative to category/recent spending, velocity steps, time of day and new-device interactions. A nearest-neighbour distance ratio compares transactions with stored fraud and legitimate training references; training excludes each row's self-match. The interaction member uses category medians from train plus unlabeled public-test features; fraud-risk mappings use training labels only.

## 3. Model Selection & Justification

Regularised logistic regression was more reliable than the tested tree and neural alternatives with only 353 positives. R2op combines logits from the engineered LR (56%), pairwise-interaction LR (24%) and threshold-basis LR, GALR (20%). The test burst rate is 3.12 times the training rate. Synthetic-looking feature/hour blocks and legitimate-burst profiles motivated fixed group adjustments. Young bursts and mature-account new devices are demoted; selected mature-account high/mid-value purchases are promoted. The T2 high-value box is shrunk toward a working 3.5% level, retaining 30% of its logit variation.

## 4. Evaluation Strategy

The historical public PR-AUC rose from 0.17206 (v3) to 0.23434 (R2op); no new public evaluation was performed. Fresh three-seed, five-fold checks give mean base-blend average precision 0.2513, or 0.2456 with extra validation missingness; these are training-distribution results before test-specific adjustments. Preprocessing and supervised references stay inside folds; unlabeled validation features supply transductive medians. Three focused interaction variants failed the predeclared confirmation gate, so the winner was retained. Fresh full training reproduces its scores within 2.8e-12. Eight isolated notebook cases pass, including shuffled ID-less data, malformed values and empty inputs.

## 5. Limitations & Risks

Public-board tuning can overfit that evaluation split, and the private distribution may differ. The generator and test labels are unavailable: the T2 rate and within-box randomness are hypotheses, not measured labels; no performance ceiling is established. A fixed +2.188389 logit shift preserves ranking and flags 300 public rows at 0.5, but changes calibration. Final outputs are operating-point-adjusted scores. The private decision threshold is unspecified, so improved private F1/recall cannot be claimed. Deployment would require representative labeled validation, probability calibration, a review budget and monitoring of false alarms and drift.
