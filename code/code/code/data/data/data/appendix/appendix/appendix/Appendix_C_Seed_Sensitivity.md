# Appendix C: Sensitivity Analysis Across Multiple Random Seeds

A subset of 300 images (10% of the full 3,000-image set, balanced across condition pairs and weight ratios) was regenerated. Five additional seeds were used: [0, 123, 456, 789, 101112]. Key compliance metrics showed low variance across seeds.

- Standard deviation of Canny compliance: 0.04
- Standard deviation of Depth compliance: 0.03
- Standard deviation of interference index: 0.05

Full results are available in `data/seed_sensitivity.csv`.
