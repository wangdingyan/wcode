import torch
import time
from tqdm import tqdm


def iterate(
        net,
        dataset,
        optimizer,
        args,
        test=False,
        save_path=None,
        pdb_ids=None,
        summary_writer=None,
        epoch_number=None,
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

    if test:
        net.eval()
        torch.set_grad_enabled(False)
    else:
        net.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    info = []
    total_processed_pairs = 0
    # Loop over one epoch:
    for it, protein_pair in enumerate(
            tqdm(dataset)
    ):  # , desc="Test " if test else "Train")):
        protein_batch_size = protein_pair.atom_coords_p1_batch[-1].item() + 1
        if save_path is not None:
            batch_ids = pdb_ids[
                        total_processed_pairs: total_processed_pairs + protein_batch_size
                        ]
            total_processed_pairs += protein_batch_size

        protein_pair.to(args.device)

        if not test:
            optimizer.zero_grad()

        # Generate the surface:
        # torch.cuda.synchronize()
        surface_time = time.time()
        P1_batch, P2_batch = process(args, protein_pair, net)
        # torch.cuda.synchronize()
        surface_time = time.time() - surface_time

        for protein_it in range(protein_batch_size):
            # torch.cuda.synchronize()
            iteration_time = time.time()

            P1 = extract_single(P1_batch, protein_it)
            P2 = None if args.single_protein else extract_single(P2_batch, protein_it)

            if args.random_rotation:
                P1["rand_rot"] = protein_pair.rand_rot1.view(-1, 3, 3)[0]
                P1["atom_center"] = protein_pair.atom_center1.view(-1, 1, 3)[0]
                P1["xyz"] = P1["xyz"] - P1["atom_center"]
                P1["xyz"] = (
                    torch.matmul(P1["rand_rot"], P1["xyz"].T).T
                ).contiguous()
                P1["normals"] = (
                    torch.matmul(P1["rand_rot"], P1["normals"].T).T
                ).contiguous()
                if not args.single_protein:
                    P2["rand_rot"] = protein_pair.rand_rot2.view(-1, 3, 3)[0]
                    P2["atom_center"] = protein_pair.atom_center2.view(-1, 1, 3)[0]
                    P2["xyz"] = P2["xyz"] - P2["atom_center"]
                    P2["xyz"] = (
                        torch.matmul(P2["rand_rot"], P2["xyz"].T).T
                    ).contiguous()
                    P2["normals"] = (
                        torch.matmul(P2["rand_rot"], P2["normals"].T).T
                    ).contiguous()
            else:
                P1["rand_rot"] = torch.eye(3, device=P1["xyz"].device)
                P1["atom_center"] = torch.zeros((1, 3), device=P1["xyz"].device)
                if not args.single_protein:
                    P2["rand_rot"] = torch.eye(3, device=P2["xyz"].device)
                    P2["atom_center"] = torch.zeros((1, 3), device=P2["xyz"].device)

            # torch.cuda.synchronize()
            prediction_time = time.time()
            outputs = net(P1, P2)
            # torch.cuda.synchronize()
            prediction_time = time.time() - prediction_time

            P1 = outputs["P1"]
            P2 = outputs["P2"]

            if args.search:
                generate_matchinglabels(args, P1, P2)

            if P1["labels"] is not None:
                loss, sampled_preds, sampled_labels = compute_loss(args, P1, P2)
            else:
                loss = torch.tensor(0.0)
                sampled_preds = None
                sampled_labels = None

            # Compute the gradient, update the model weights:
            if not test:
                # torch.cuda.synchronize()
                back_time = time.time()
                loss.backward()
                optimizer.step()
                # torch.cuda.synchronize()
                back_time = time.time() - back_time

            if it == protein_it == 0 and not test:
                for para_it, parameter in enumerate(net.atomnet.parameters()):
                    if parameter.requires_grad:
                        summary_writer.add_histogram(
                            f"Gradients/Atomnet/para_{para_it}_{parameter.shape}",
                            parameter.grad.view(-1),
                            epoch_number,
                        )
                for para_it, parameter in enumerate(net.conv.parameters()):
                    if parameter.requires_grad:
                        summary_writer.add_histogram(
                            f"Gradients/Conv/para_{para_it}_{parameter.shape}",
                            parameter.grad.view(-1),
                            epoch_number,
                        )

                for d, features in enumerate(P1["input_features"].T):
                    summary_writer.add_histogram(f"Input features/{d}", features)

            if save_path is not None:
                save_protein_batch_single(
                    batch_ids[protein_it], P1, save_path, pdb_idx=1
                )
                if not args.single_protein:
                    save_protein_batch_single(
                        batch_ids[protein_it], P2, save_path, pdb_idx=2
                    )

            try:
                if sampled_labels is not None:
                    roc_auc = roc_auc_score(
                        np.rint(numpy(sampled_labels.view(-1))),
                        numpy(sampled_preds.view(-1)),
                    )
                else:
                    roc_auc = 0.0
            except Exception as e:
                print("Problem with computing roc-auc")
                print(e)
                continue

            R_values = outputs["R_values"]

            info.append(
                dict(
                    {
                        "Loss": loss.item(),
                        "ROC-AUC": roc_auc,
                        "conv_time": outputs["conv_time"],
                        "memory_usage": outputs["memory_usage"],
                    },
                    # Merge the "R_values" dict into "info", with a prefix:
                    **{"R_values/" + k: v for k, v in R_values.items()},
                )
            )
            # torch.cuda.synchronize()
            iteration_time = time.time() - iteration_time

    # Turn a list of dicts into a dict of lists:
    newdict = {}
    for k, v in [(key, d[key]) for d in info for key in d]:
        if k not in newdict:
            newdict[k] = [v]
        else:
            newdict[k].append(v)
    info = newdict

    # Final post-processing:
    return info


def process(args, protein_pair, net):
    P1 = process_single(protein_pair, chain_idx=1)
    if not "gen_xyz_p1" in protein_pair.keys:
        net.preprocess_surface(P1)
        #if P1["mesh_labels"] is not None:
        #    project_iface_labels(P1)
    P2 = None
    if not args.single_protein:
        P2 = process_single(protein_pair, chain_idx=2)
        if not "gen_xyz_p2" in protein_pair.keys:
            net.preprocess_surface(P2)
            #if P2["mesh_labels"] is not None:
            #    project_iface_labels(P2)

    return P1, P2

def process_single(protein_pair, chain_idx=1):
    """Turn the PyG data object into a dict."""

    P = {}
    with_mesh = "face_p1" in protein_pair.keys
    preprocessed = "gen_xyz_p1" in protein_pair.keys

    if chain_idx == 1:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p1 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p1_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p1 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p1 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p1
        P["batch_atoms"] = protein_pair.atom_coords_p1_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p1
        P["atomtypes"] = protein_pair.atom_types_p1

        P["xyz"] = protein_pair.gen_xyz_p1 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p1 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p1 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p1 if preprocessed else None

    elif chain_idx == 2:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p2 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p2_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p2 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p2 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p2
        P["batch_atoms"] = protein_pair.atom_coords_p2_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p2
        P["atomtypes"] = protein_pair.atom_types_p2

        P["xyz"] = protein_pair.gen_xyz_p2 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p2 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p2 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p2 if preprocessed else None

    return P