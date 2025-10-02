wandb enabled
python -m main +name=infer \
    experiment.tasks=[validation] \
    dataset.validation_multiplier=1 \
    +diffusion_model_path=yslan/worldmem_checkpoints/diffusion_only.ckpt \
    +vae_path=yslan/worldmem_checkpoints/vae_only.ckpt \
    +customized_load=true \
    +seperate_load=true \
    dataset.n_frames=8 \
    dataset.save_dir=data/minecraft \
    +dataset.n_frames_valid=16 \
    +dataset.memory_condition_length=8 \
    +dataset.customized_validation=true \
    +dataset.add_timestamp_embedding=true \
    +dataset.selection_mode=list \
    +dataset.selected_videos='["wo_updown_000001","wo_updown_000005"]'\
    +algorithm.n_tokens=8 \
    +algorithm.memory_condition_length=8 \
    algorithm.context_frames=8 \
    +algorithm.relative_embedding=true \
    +algorithm.log_video=true \
    +algorithm.add_timestamp_embedding=true \
    algorithm.condition_index_method=fov \
    algorithm.metrics=[lpips,psnr,fid] \
    +infer_script_path=infer.sh \

# +dataset.selection_mode=random or +dataset.selection_mode=list \ +dataset.selected_videos=[w_updown_000008]