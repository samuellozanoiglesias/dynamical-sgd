import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import pinvh
from scipy.sparse.linalg import ArpackError, svds
from tqdm import tqdm


def compute_class_weights(t, focus_class, w_max, period_length, num_classes, device):
    slope = 2 * (w_max - 1) / period_length
    if t < period_length / 2:
        w_main = 1 + t * slope
    else:
        w_main = 2 * w_max - t * slope - 1
    weights = np.ones(num_classes, dtype=np.float32)
    weights[focus_class] = float(w_main)
    weights = weights / np.sum(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_epoch(
    model,
    criterion,
    device,
    num_classes,
    train_loader,
    optimizer,
    batch_size,
    global_step,
    apply_bumping,
    bump_period_length,
    bump_w_max,
    max_steps_this_epoch,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_count = 0
    processed_batches = 0
    class_focus = None

    train_iter = iter(train_loader)
    while processed_batches < max_steps_this_epoch:
        try:
            data, target = next(train_iter)
        except StopIteration:
            # Restart loader so each epoch can always hit the requested step count.
            train_iter = iter(train_loader)
            data, target = next(train_iter)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)

        if str(criterion) == "CrossEntropyLoss()":
            if apply_bumping:
                class_focus = (global_step // bump_period_length) % num_classes
                class_weights = compute_class_weights(global_step % bump_period_length, class_focus, bump_w_max, bump_period_length, num_classes, device)
                sample_weights = class_weights[target]
                per_sample_loss = F.cross_entropy(out, target, reduction="none")
                loss = torch.mean(per_sample_loss * sample_weights)
            else:
                loss = criterion(out, target)
        else:
            target_one_hot = F.one_hot(target, num_classes=num_classes).float()
            loss = criterion(out, target_one_hot)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(out, dim=1)
        running_correct += int(torch.sum(preds == target).item())
        running_count += int(target.shape[0])
        running_loss += float(loss.item())
        processed_batches += 1
        global_step += 1

    mean_loss = running_loss / max(1, processed_batches)
    epoch_accuracy = running_correct / max(1, running_count)
    return mean_loss, epoch_accuracy, global_step, class_focus, processed_batches


@torch.no_grad()
def evaluate(model, criterion, device, num_classes, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        out = model(data)

        if str(criterion) == "CrossEntropyLoss()":
            loss = F.cross_entropy(out, target, reduction="sum")
        else:
            target_one_hot = F.one_hot(target, num_classes=num_classes).float()
            loss = F.mse_loss(out, target_one_hot, reduction="sum")

        total_loss += float(loss.item())
        total_correct += int(torch.sum(torch.argmax(out, dim=1) == target).item())
        total_count += int(target.shape[0])

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


@torch.no_grad()
def analysis(graphs, model, criterion_summed, device, num_classes, loader, feature_store, classifier, weight_decay, loss_name):
    model.eval()

    n_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)
    mean_sum = None
    sw = None

    loss = 0.0
    net_correct = 0
    ncc_match_net = 0

    for computation in ["Mean", "Cov"]:
        pbar = tqdm(total=len(loader), position=0, leave=True, disable=True)
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            h = feature_store.value.data.view(data.shape[0], -1)
            if mean_sum is None:
                feat_dim = h.shape[1]
                mean_sum = torch.zeros(num_classes, feat_dim, dtype=h.dtype, device=device)
                sw = torch.zeros(feat_dim, feat_dim, dtype=h.dtype, device=device)

            if computation == "Mean":
                if str(criterion_summed) == "CrossEntropyLoss()":
                    loss += criterion_summed(output, target).item()
                else:
                    loss += criterion_summed(output, F.one_hot(target, num_classes=num_classes).float()).item()

                mean_sum.index_add_(0, target, h)
                n_per_class += torch.bincount(target, minlength=num_classes)
            else:
                class_means = m.T[target]
                z = h - class_means
                sw += z.T @ z

                net_pred = torch.argmax(output, dim=1)
                net_correct += int(torch.sum(net_pred == target).item())

                ncc_scores = torch.cdist(h, m.T)
                ncc_pred = torch.argmin(ncc_scores, dim=1)
                ncc_match_net += int(torch.sum(ncc_pred == net_pred).item())

            pbar.update(1)
        pbar.close()

        if computation == "Mean":
            mean = torch.zeros_like(mean_sum)
            valid = n_per_class > 0
            mean[valid] = mean_sum[valid] / n_per_class[valid].unsqueeze(1)
            m = mean.T
            loss /= max(1, int(torch.sum(n_per_class).item()))
        else:
            sw /= max(1, int(torch.sum(n_per_class).item()))

    graphs.loss.append(loss)
    total_count = max(1, int(torch.sum(n_per_class).item()))
    graphs.accuracy.append(net_correct / total_count)
    graphs.NCC_mismatch.append(1 - ncc_match_net / total_count)

    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param**2).item()
    graphs.reg_loss.append(reg_loss)

    mu_g = torch.mean(m, dim=1, keepdim=True)
    m_ = m - mu_g
    sb = torch.matmul(m_, m_.T) / num_classes

    w = classifier.weight
    m_norms = torch.norm(m_, dim=0)
    w_norms = torch.norm(w.T, dim=0)
    eps = 1e-12

    graphs.mu_c_norm_avg.append(torch.mean(m_norms).item())
    graphs.mu_G_norm.append(torch.norm(mu_g).item())
    graphs.M_fro_norm.append(torch.norm(m_, p="fro").item())
    graphs.W_fro_norm.append(torch.norm(w, p="fro").item())

    graphs.norm_M_CoV.append((torch.std(m_norms) / torch.clamp(torch.mean(m_norms), min=eps)).item())
    graphs.norm_W_CoV.append((torch.std(w_norms) / torch.clamp(torch.mean(w_norms), min=eps)).item())

    sw_np = sw.cpu().numpy()
    sb_np = sb.cpu().numpy()
    try:
        eigvec, eigval, _ = svds(sb_np, k=max(1, num_classes - 1))
        safe_inv = np.where(np.abs(eigval) > eps, 1.0 / eigval, 0.0)
        inv_sb = eigvec @ np.diag(safe_inv) @ eigvec.T
    except (ArpackError, ValueError, np.linalg.LinAlgError):
        # Early in training Sb can be numerically rank-deficient; use stable pseudo-inverse.
        inv_sb = pinvh(sb_np, atol=eps)
    graphs.Sw_invSb.append(float(np.trace(sw_np @ inv_sb)))

    normalized_m = m_ / torch.clamp(torch.norm(m_, "fro"), min=eps)
    normalized_w = w.T / torch.clamp(torch.norm(w.T, "fro"), min=eps)
    graphs.W_M_dist.append((torch.norm(normalized_w - normalized_m) ** 2).item())

    def eq_stats(v):
        g = v.T @ v
        mask = ~torch.eye(num_classes, dtype=torch.bool, device=device)
        off = g[mask]
        target = -1.0 / (num_classes - 1)
        return torch.std(off).item(), torch.mean(torch.abs(off - target)).item()

    cos_m_std, cos_m_mean = eq_stats(m_ / torch.clamp(m_norms, min=eps))
    cos_w_std, cos_w_mean = eq_stats(w.T / torch.clamp(w_norms, min=eps))

    graphs.cos_M_std.append(cos_m_std)
    graphs.cos_W_std.append(cos_w_std)
    graphs.cos_M.append(cos_m_mean)
    graphs.cos_W.append(cos_w_mean)
