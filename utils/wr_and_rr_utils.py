import numpy as np
import json
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from matplotlib.colors import ListedColormap


def get_white_viridis_cmap(range_start=96, range_end=196):
    """Colormap for the heatmaps."""

    viridis = plt.get_cmap("viridis", 256)
    viridis_colors = viridis.colors.copy()

    viridis_white = viridis_colors[range_end + 1].copy()

    viridis_colors[0:range_start, 0] = np.ones(range_start)
    viridis_colors[0:range_start, 1] = np.ones(range_start)
    viridis_colors[0:range_start, 2] = np.ones(range_start)
    viridis_colors[0:range_start, 3] = np.ones(range_start)

    viridis_colors[range_start:range_end, 0] = np.linspace(
        1, viridis_white[0], range_end - range_start
    )
    viridis_colors[range_start:range_end, 1] = np.linspace(
        1, viridis_white[1], range_end - range_start
    )
    viridis_colors[range_start:range_end, 2] = np.linspace(
        1, viridis_white[2], range_end - range_start
    )
    viridis_colors[range_start:range_end, 3] = np.linspace(
        1, viridis_white[3], range_end - range_start
    )

    newcmp = ListedColormap(viridis_colors)
    return newcmp


def show_explanation_single(
    img, mask, method="heat_map", sign="positive", save_name=None
):
    fig, ax = viz.visualize_image_attr(
        mask[0],
        img[0],
        method=method,
        cmap=get_white_viridis_cmap(0, 128),
        sign=sign,
        show_colorbar=True,
        use_pyplot=False,
        plt_fig_axis=(fig, ax),
        alpha_overlay=1.0,
    )

    if save_name is not None:
        name_ = "eval_output/output_images/" + save_name + ".png"
        plt.savefig(name_, bbox_inches="tight")
        plt.close()

        # # save img
        # name_raw_ = "output_images/raw/" + save_name + ".png"
        # # create directory if not exists
        # os.makedirs(os.path.dirname(name_raw_), exist_ok=True)

        # mean = np.asarray([0.485, 0.456, 0.406])
        # std = np.asarray([0.229, 0.224, 0.225])

        # # Revert normalization
        # img_np = img[0]
        # img_reverted = (img_np * std + mean) * 255.0

        # # Clip values to the valid range [0, 255] and convert to uint8
        # img_reverted = np.clip(img_reverted, 0, 255).astype(np.uint8)

        # plt.imsave(name_raw_, img_reverted)

    else:
        plt.show()


def show_explanation_overlay_grid_captum(
    imgs,
    masks,
    titles,
    figsize=(12, 6),
    n_col=4,
    method="blended_heat_map",
    sign="positive",
    fig_header="?",
    save_name=None,
):
    """
    Show heatmaps.

    Args:
        imgs: list of numpy arrays.
        masks: list of numpy arrays.
        titles: list of strings.
        figsize: sets the size of the figure.
        n_col: sets the number of columns in the plot.
        method: specify the method with wihich the plot is generated (same as in captum library).
            'blended_heat_map' -> lay heatmap on top of image.
            'heat_map'         -> only plot heatmap.
            'original_image'   -> only display original image.
            'masked_image'  -> Mask image (pixel-wise multiply) by normalized attribution values.
            'alpha_scaling' -> Sets alpha channel of each pixel to be equal
                to normalized attribution value.
        sign: chosen sign of attributions to visualize. Supported options are:
            'positive' -> displays only positive pixel attributions.
            'absolute_value' -> displays absolute value of attributions.
            'negative' -> displays only negative pixel attributions.
        fig_header: haeder of the generated plot.
    """

    n_row = int(np.ceil(len(imgs) / n_col))
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    fig.suptitle("Explanation generated with " + fig_header, fontsize=13)
    fig.subplots_adjust(hspace=0.4)

    axs = axs.flatten()
    for img, mask, title, ax in zip(imgs, masks, titles, axs):
        fig, ax = viz.visualize_image_attr(
            mask,
            img,
            method=method,
            sign=sign,
            show_colorbar=True,
            title=title,
            cmap=get_white_viridis_cmap(0, 128),
            alpha_overlay=0.7,
            use_pyplot=False,
            plt_fig_axis=(fig, ax),
        )

    if save_name is not None:
        name_ = save_name + ".png"
        plt.savefig(name_, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def show_explanation_overlay_grid_captum_2(
    imgs,
    masks,
    titles,
    figsize=(6, 7),
    n_col=2,
    method="heat_map",
    sign="positive",
    fig_header="?",
    save_name=None,
):
    """Show image and heatmaps next to each other."""

    # n_row = int(np.ceil(len(imgs) / n_col)*2)
    n_row = len(imgs)
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    fig.suptitle("Explanation generated with " + fig_header, fontsize=13)
    fig.subplots_adjust(hspace=0.4)

    axs = axs.flatten()
    is_img = 0
    img_counter = 0
    mask_counter = 0
    for ax in axs:
        if is_img == 0:
            if img_counter >= len(imgs):
                break
            ax.imshow(imgs[img_counter], cmap="gray", interpolation="none")
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            is_img = 1
            img_counter += 1
        else:
            fig, ax = viz.visualize_image_attr(
                masks[mask_counter],
                imgs[mask_counter],
                method=method,
                sign=sign,
                show_colorbar=True,
                title=titles[mask_counter],
                cmap=get_white_viridis_cmap(0, 128),
                use_pyplot=False,
                plt_fig_axis=(fig, ax),
            )
            is_img = 0
            mask_counter += 1

    if save_name is not None:
        name_ = save_name + ".png"
        plt.savefig(name_, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def show_img_grid_numpy(imgs, figsize=(10, 4), n_col=4, save_name=None):

    n_row = int(np.ceil(len(imgs) / n_col))
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    fig.suptitle("Orginial images", fontsize=13)
    fig.subplots_adjust(hspace=0.2)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img, cmap="gray", interpolation="none")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if save_name is not None:
        name_ = save_name + ".png"
        plt.savefig(name_, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def explain_with_ig_one_by_one(
    model,
    dataloader,
    sign="positive",
    fig_header="IG (Ross)",
    next_to_each_other=False,
    save_name=None,
    device="cuda",
    clip_pixel_values_to_zero_one=True,
    specified_img_indices=[],
):
    """
    Explain (visualize) the predictions of the model using Input Gradients
    according to Ross et. al (2017). One image per plot.
    Args:
        model: the pytorch model.
        dataloader: pytorch dataloader (images, labels, _ ).
        sign: specifies which attributions should be visualized. Either 'positive', 'negative',
            'absolute_value', 'all'. (see captum library viz.visualize_image_attr()).
        fig_header: title of the generated image.
        next_to_each_other: if True then plots one figure with the original image left and the
            heatmap on the right side. If False then plot two seperate plots for org. images
            and corresponding heatmaps.
        gray_images: set to True if input are gray images (MNIST)
        save_name: if specified the resulting plot will be saved to output_images folder and not
            be printed.
        clip_pixel_values_to_zero_one: Squeeze pixels values to range [0,1] (used for RGB images)
        device: either 'cpu' or 'cuda'
        specified_img_indices: list of indices for the images you want to explain. Takes the specified
            indices from the first batch of the dataloader (batchsize needs to be bigger
            than the lenght of the specified indices.

    Returns:
        Visualizes the images and the corresponding explanation heatmaps to two or one
            plt figures (best used in Notebooks).
    """
    model.eval()
    img_id = 0

    for data in dataloader:
        images_t, labels_t = data[0], data[1]
        for i in range(len(labels_t)):
            if len(specified_img_indices) == 0 or img_id in specified_img_indices:

                model.zero_grad()
                original_image = np.transpose(
                    (images_t[i].cpu().detach().numpy()), (1, 2, 0)
                )
                img = images_t[i].unsqueeze(0).to(device)
                gt = labels_t[i]
                img.requires_grad_()

                logit = model(img)

                _, predicted = torch.max(F.softmax(logit, dim=1), 1)

                log_prob_ys = F.log_softmax(logit, dim=1)
                log_prob_ys.retain_grad()
                ig = torch.autograd.grad(
                    log_prob_ys,
                    img,
                    torch.ones_like(log_prob_ys),
                    create_graph=True,
                    allow_unused=True,
                )[0]

                if ig.shape[1] == 1:  # image with one channel (black white image)
                    attr = ig.squeeze().unsqueeze(2).cpu().detach().numpy()
                else:  # color image
                    attr = np.transpose(ig.squeeze().detach().cpu().numpy(), (1, 2, 0))

                title = "y=" + str(gt.item()) + " pred=" + str(predicted.item())

                if np.sum(attr) == 0.0:
                    print(
                        f"Attribution mask is zero. Could not generate attribution with ig for image at index {i}."
                    )
                    img_id += 1
                    continue

                if clip_pixel_values_to_zero_one:
                    original_image = np.clip(original_image, 0, 1)

                imgs, attrs, titles = [original_image], [attr], [title]

                cur_save_name = save_name + "-" + str(img_id)

                if next_to_each_other:
                    show_explanation_single(
                        imgs,
                        attrs,
                        method="heat_map",
                        sign="positive",
                        save_name=cur_save_name,
                    )
                    # util.show_explanation_overlay_grid_captum_2(imgs, attrs, titles, sign=sign, \
                    #    fig_header=fig_header, method='heat_map', save_name=cur_save_name, figsize=(6,4))

                else:
                    show_img_grid_numpy(imgs, save_name=cur_save_name)
                    show_explanation_overlay_grid_captum(
                        imgs,
                        attrs,
                        titles,
                        sign=sign,
                        fig_header=fig_header,
                        method="heat_map",
                        save_name=cur_save_name,
                    )

                print(f"explanation image with name {cur_save_name} saved!")

                img_id += 1

            else:
                img_id += 1
                continue


def norm_saliencies_fast(A, positive_only=False):
    """
    Normalize tensor to [0,1] across first dimension (for every batch_i) according to formula
    t(i) = (i_t - min_t) /(max_t + 1e-6).
    Add small constant to prevent zero divison.

    Args:
        A: tensor of shape (n, c, h, w).
        positive_only: if True then take only positive values into account and
            zero out negative values.
    """
    shape = A.shape
    A = A.view(A.size(0), -1)
    if positive_only:
        A[A < 0] = 0.0
    A -= A.min(1, keepdim=True)[0]
    A /= A.max(1, keepdim=True)[0] + 1e-6  # add small constant preventing zero divison
    A = A.view(shape)
    return A


def quantify_wr_rr_reason(
    dataloader,
    model,
    device,
    foldername,
    metric,
    threshold=None,
    mode="mean",
    flags=True,
):
    """
    Quantifies wrong reason based on ground-truth explanations (i.e.
    measures the confounder activation for whole test set). Either mean or
    median pixel-level activation. If threshold=None then calculate
    the binarization threshold. To binarize heatmaps before measurement insert
    threshold.

    Args:
        dataloader: pytorch dataloader (images, _, expl, flags) where expl are the
            ground-truth masks.
        model: the pytorch model.
        device: either 'cpu' or 'cuda'
        name: name of the model or identifier
        foldername: name of the output folder
        metric: either 'wr' or 'rr'
        threshold: if None, then calculates the mean or median avg activation
            binarization threshold.
        mode: either 'mean' or 'median'. Mean calculates the mean pixel-level activation
            averaged over the whole dataloader. Median calculates the median pixel-level
            activation and returns the median of the whole dataloader.
        flags: it True then only images with flag=1 are taking into account (ISIC19).

    """

    model.eval()
    actScores = []
    number_instances = 0
    cannot_attribute_num = 0

    with tqdm(dataloader, unit="batch") as tbatch:
        for data in tbatch:
            mask_index = 2 if metric == "wr" else 3
            if flags:
                images_t, masks_t, flags_t = (
                    data[0].to(device),
                    data[mask_index].to(device),
                    data[6].to(device),
                )
            else:
                images_t, masks_t = data[0].to(device), data[mask_index].to(device)

            images_t.requires_grad_()
            logits = model(images_t)
            h, w = images_t.shape[2], images_t.shape[3]

            model.zero_grad()
            # get gradients w.r.t. to the input
            log_prob_ys = F.log_softmax(logits, dim=1)
            log_prob_ys.retain_grad()
            attr = torch.autograd.grad(
                log_prob_ys,
                images_t,
                torch.ones_like(log_prob_ys),
                create_graph=True,
                allow_unused=True,
            )[0]
            # attr.shape => (n,3,h,w)
            # sum over rgb channels
            attr = torch.sum(attr, dim=1).unsqueeze(1)

            norm_attr = norm_saliencies_fast(attr, positive_only=True)
            # filter out instances that have no seg mask
            if flags:
                have_confounder_indices = torch.nonzero((flags_t == 1), as_tuple=True)[
                    0
                ]
                norm_attr = norm_attr[have_confounder_indices]
                masks_t = masks_t[have_confounder_indices]
                tmp = len(have_confounder_indices)
            else:
                tmp = len(images_t)
            # filter out attr which are zero --> assuming no attr can be calculated
            sums = torch.sum(norm_attr, dim=(1, 2, 3))
            only_non_zero_attr_indices = torch.nonzero(sums, as_tuple=True)[0]
            norm_attr = norm_attr[only_non_zero_attr_indices]
            masks_t = masks_t[only_non_zero_attr_indices]
            # flags_t = flags_t[only_non_zero_attr_indices]
            cannot_attribute_num += tmp - len(only_non_zero_attr_indices)
            number_instances += len(only_non_zero_attr_indices)

            if norm_attr.size(0) == 0:
                continue

            if threshold is not None:
                norm_attr[norm_attr > threshold] = 1.0
                norm_attr[norm_attr < threshold] = 0.0

                attr_x_expl = torch.mul(masks_t, norm_attr)
                flat_attr_x_expl = attr_x_expl.view(attr_x_expl.size(0), -1)
                attr_ca = torch.sum(flat_attr_x_expl, dim=1)
                masks_flat = masks_t.view(masks_t.size(0), -1)
                attr_max = torch.sum(masks_flat, dim=1)

                print(torch.unique(attr_max))

                actScore = torch.div(attr_ca, attr_max)
                actScores += actScore.tolist()

            else:  # calc median/mean
                if mode == "mean":
                    scores = torch.div(
                        torch.sum(norm_attr.view(norm_attr.size(0), -1), dim=1), (h * w)
                    )
                elif mode == "median":
                    scores = torch.median(norm_attr.view(norm_attr.size(0), -1), dim=1)[
                        0
                    ]
                actScores += scores.tolist()

    if number_instances != len(actScores):
        raise RuntimeWarning(
            f"Counted instances which have seg mask ({number_instances}) not equal to length of actScores ({len(actScores)})!!!"
        )

    if threshold is not None:
        abs_activation = sum(actScores)
        avg_activation_per_instance = np.mean(actScores)
        std = np.std(actScores)
        # print(f"ActScores: {actScores}")
        print(f"Number of actScores= {len(actScores)}")
        print(f"Activation AVG per instance = {100*avg_activation_per_instance}")
        print(f"STD = {100*std}")
        print(f"Activation ABS sum = {abs_activation}")
        print(f"Number of complete zero attr= {cannot_attribute_num}")
        data = {
            "avg_activation_per_instance": 100 * avg_activation_per_instance,
            "std_activation_per_instance": 100 * std,
            "abs_activation": abs_activation,
            "length": number_instances,
            "cannot_attribute_num": cannot_attribute_num,
            "actScores": actScores,
        }
        print(foldername)
        os.makedirs(foldername, exist_ok=True)
        with open(
            foldername + f"-{mode}_{metric}_stats.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return 100 * avg_activation_per_instance

    else:
        if mode == "mean":
            value = np.mean(actScores)
        elif mode == "median":
            value = np.median(actScores)  # get median of list of medians

        print(f"{str(mode)}= {float(value)}")

        print(f"Number of actScores= {len(actScores)}")
        print(f"Number of complete zero attr= {cannot_attribute_num}")
        data = {
            str(mode): float(value),
            "num": number_instances,
            "cannot_attribute_num": cannot_attribute_num,
            "actScores": actScores,
        }

        os.makedirs(foldername, exist_ok=True)
        with open(
            foldername + f"-{mode}-{metric}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return float(value)  # , len(actScores)
