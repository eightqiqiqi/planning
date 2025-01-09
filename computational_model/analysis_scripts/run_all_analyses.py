# in this script, we call all of the model analysis functions.
# this may take a while to run unless you have a very big computer

import time

print("running all analyses")
tic = time.time()

# Include and execute all analysis scripts
exec(open("analyse_human_data.py").read())
exec(open("calc_human_prior.py").read())

exec(open("analyse_rollout_timing.py").read())
exec(open("repeat_human_actions.py").read())
exec(open("perf_by_rollout_number.py").read())
exec(open("compare_perf_without_rollout.py").read())
exec(open("shuffle_rollout_times.py").read())
exec(open("behaviour_by_success.py").read())
exec(open("model_replay_analyses.py").read())
exec(open("rollout_as_pg.py").read())
# exec(open("analyse_by_N.py").read())  # Uncomment if pretrained models are included
exec(open("estimate_num_mazes.py").read())

exec(open("compare_maze_path_lengths.py").read())
# exec(open("analyse_hp_sweep.py").read())  # Uncomment if pretrained models are included
exec(open("eval_value_function.py").read())
# exec(open("quantify_internal_model.py").read())  # Uncomment if full training run or pretrained models are included
# exec(open("analyse_variable_rollouts.py").read())  # Uncomment if pretrained models are included

print("\nFinished after ", (time.time() - tic) / 60 / 60, " hours.")
