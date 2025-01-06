# bin/bash
python -u -m habitat_baselines.run \
    --config-name=single_rearrange/test_fetch.yaml \
    habitat.dataset.data_path=data/datasets/sat_DLC_test10_dataset/image/80/scene_graph.gz \
    habitat.seed=42 \
    habitat_baselines.image_dir="video_dir_test_pivot" \
    habitat_baselines.video_dir="video_dir_test_pivot" \
#运行机制：habitat_mas_evaluator中调用agent.actor_critic.act执行，此处方法是HierarchicalPolicy的方法
#HierarchicalPolicy的方法act中_update_skills()，分别对上下层的skill都进行更新
#_update_skills()方法中self._high_level_policy.get_next_skill()调用着hl_xxx
#方法（对应xxx_policy.py）中
#现在想的是用llm_policy中的方法，然后在get_next_skill中判断是agent_0或者agent_1，
#分配对应的"target_obj"，在OracleNavPolicy作符号判断，就可以传标志位给OracleNavDiffAction

# python -u -m habitat_baselines.run --config-name=multi_rearrange/zxz_llm_fetch_stretch_man.yaml

# {'pick': 0, 'place': 1, 'wait': 2, 'nav_to_obj': 3, 'nav_to_position': 4, 'reset_arm': 5, 'turn_left': 6, 'turn_right': 7, 'move_forward': 8, 'move_backward': 9}