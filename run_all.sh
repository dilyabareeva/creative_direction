python evaluation/evaluate_before_intervention.py

python train_intervention.py --alpha 0.2 --p_type "add"
python train_intervention.py --alpha 0.2 --p_type "affine"

python evaluation/evaluate_after_intervention.py --alpha 0.2 --p_type "add"
python evaluation/evaluate_after_intervention.py --alpha 0.2 --p_type "affine"

python evaluation/evaluate_transitions.py