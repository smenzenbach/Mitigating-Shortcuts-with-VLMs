import logging
import time

import h5py
import numpy as np
from sympy import use


def get_dataloaders(
    batch_size=16,
    train_shuffle=True,
    number_C_class=None,
    number_NC_class=None,
    use_inpainted=False,
    use_gt_masks=True,
    id_tag=None,
    only_train_C=False,
    only_train_NC=False,
    rr_masks_available=False,
    raw_path="/workspace/home/data/Pascal_VOC07/h5_files/",
    pipeline_output_path=None,
    target_label={"confounded": 1, "non-confounded": 0},
):
    """
    Load dataset and return train and test set Dataloaders.
    c: confounded class, nc: non-confounded class

    Parameters
    ----------
    batch_size : int, optional
        Batch size for the Dataloaders. The default is 16.
    train_shuffle : bool, optional
        sSts the pytorch Dataloader attribute 'shuffle' which
                'have the data reshuffled at every epoch'. The default is True.
    number_C_class : int, optional
        Number of samples of confounded class to load. The default is None (all samples are loaded).
    number_NC_Class : int, optional
        Number of samples of non-confounded class to load. The default is None (all samples are loaded).
    use_inpainted : bool, optional
        If True, the inpainted images are used for training. The default is False.
    use_gt_masks : bool, optional
        If True, the ground truth masks are used. The default is True.
    id_tag : str, optional
        Tag to identify the configuration. The default is None.
    only_train_C : bool, optional
        If True, only confounded samples of the confounded class plus the non-confounded class are loaded. The default is False.
    only_train_NC : bool, optional
        If True, only non-confounded samples of the confounded class plus the non-confounded class are loaded. The default is False.
    rr_masks_available : bool, optional
        If True, the masks for the inpainted images are used. The default is False.
    raw_path : str, optional
        Path to the raw data. The default is "/workspace/home/data/Pascal_VOC07/h5_files/".
    pipeline_output_path : str, optional
        Path to the output of the pipeline data. The default is None.
    target_label : dict, optional
        Dictionary with the target labels. The default is {"confounded": 1, "non-confounded": 0}.
    """

    print("\n----------Dataset----------")
    logging.info("----------Dataset----------")

    if only_train_C and only_train_NC:
        raise ValueError(
            "only_train_C and only_train_NC cannot be set to True at the same time."
        )

    start = time.time()
    print("Read in data from .h5 files...")
    logging.info("Read in data from .h5 files...")
    # Load confounded class
    # imgs
    with h5py.File(raw_path + "cc_imgs.h5", "r") as hf:
        if number_C_class is None:
            cc_imgs = hf["cc_imgs"][:]
        else:
            cc_imgs = hf["cc_imgs"][:number_C_class]
    # flags
    with h5py.File(raw_path + "cc_flags.h5", "r") as hf:
        if number_C_class is None:
            cc_flags = hf["cc_flags"][:]
        else:
            cc_flags = hf["cc_flags"][:number_C_class]
    # masks
    if use_gt_masks:
        with h5py.File(raw_path + "cc_masks.h5", "r") as hf:
            if number_C_class is None:
                cc_masks = hf["cc_masks"][:]
            else:
                cc_masks = hf["cc_masks"][:number_C_class]
    else:
        with h5py.File(pipeline_output_path + "cc_masks.h5", "r") as hf:
            if number_C_class is None:
                cc_masks = hf["cc_masks"][:]
            else:
                cc_masks = hf["cc_masks"][:number_C_class]
    # inpainted images
    if use_inpainted:
        with h5py.File(pipeline_output_path + "cc_inpainted_imgs.h5", "r") as hf:
            if number_C_class is None:
                cc_inpainted_imgs = hf["cc_inpainted_imgs"][:]
            else:
                cc_inpainted_imgs = hf["cc_inpainted_imgs"][:number_C_class]
    # rr masks
    if rr_masks_available:
        with h5py.File(pipeline_output_path + "cc_rr_masks.h5", "r") as hf:
            if number_C_class is None:
                cc_rr_masks = hf["cc_rr_masks"][:]
            else:
                cc_rr_masks = hf["cc_rr_masks"][:number_C_class]
    # Load non-confounded class
    # imgs
    with h5py.File(raw_path + "nc_imgs.h5", "r") as hf:
        if number_NC_class is None:
            nc_imgs = hf["nc_imgs"][:]
        else:
            nc_imgs = hf["nc_imgs"][:number_NC_class]
    # flags
    with h5py.File(raw_path + "nc_flags.h5", "r") as hf:
        if number_NC_class is None:
            nc_flags = hf["nc_flags"][:]
        else:
            nc_flags = hf["nc_flags"][:number_NC_class]
    # masks
    if use_gt_masks:
        with h5py.File(raw_path + "nc_masks.h5", "r") as hf:
            if number_NC_class is None:
                nc_masks = hf["nc_masks"][:]
            else:
                nc_masks = hf["nc_masks"][:number_NC_class]
    else:
        with h5py.File(pipeline_output_path + "nc_masks.h5", "r") as hf:
            if number_NC_class is None:
                nc_masks = hf["nc_masks"][:]
            else:
                nc_masks = hf["nc_masks"][:number_NC_class]
    # inpainted images
    if use_inpainted:
        with h5py.File(pipeline_output_path + "nc_inpainted_imgs.h5", "r") as hf:
            if number_NC_class is None:
                nc_inpainted_imgs = hf["nc_inpainted_imgs"][:]
            else:
                nc_inpainted_imgs = hf["nc_inpainted_imgs"][:number_NC_class]
    # rr masks
    if rr_masks_available:
        with h5py.File(pipeline_output_path + "nc_rr_masks.h5", "r") as hf:
            if number_NC_class is None:
                nc_rr_masks = hf["nc_rr_masks"][:]
            else:
                nc_rr_masks = hf["nc_rr_masks"][:number_NC_class]

    end = time.time()
    elap = int(end - start)
    print(f"  --> Read in finished: Took {elap} sec!")
    logging.info(f"  --> Read in finished: Took {elap} sec!")

    # generate target labels
    cc_targets = np.full(cc_imgs.shape[0], target_label["confounded"], dtype=np.int64)
    nc_targets = np.full(
        nc_imgs.shape[0], target_label["non-confounded"], dtype=np.int64
    )

    # generate datasets
