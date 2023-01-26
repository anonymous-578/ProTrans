from typing import Dict

import torch
from torchvision.models.feature_extraction import create_feature_extractor


@torch.no_grad()
def get_features_labels(model, loader, features_dict: Dict[str, str], flatten):
    # save initial state and change the model to eval mode
    mode = model.training
    model.eval()

    # create feature extractor
    feature_extractor = create_feature_extractor(model, return_nodes=features_dict)
    features_total = {layer: [] for layer in features_dict.keys()}
    labels_total = []

    for idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        # extract features for mini-batch
        features = feature_extractor(inputs)
        if flatten:
            features = [torch.flatten(features[name], start_dim=1) for name in features_dict.values()]
        else:
            features = [features[name] for name in features_dict.values()]

        labels_total.append(labels)
        for i, layer in enumerate(features_dict.keys()):
            features_total[layer].append(features[i])

    # concat mini-batch features and labels
    for layer, features in features_total.items():
        features_total[layer] = torch.concat(features, dim=0)
    labels_total = torch.concat(labels_total, dim=0)

    # change to the saved initial state
    model.training = mode

    return features_total, labels_total


@torch.no_grad()
def get_mean_cov(features: torch.Tensor, labels: torch.Tensor, num_classes: int, sample_proto: bool, alpha: float = 0.0001):
    classes, indices = torch.unique(labels, sorted=True, return_inverse=True)
    feature_dim = features.size(-1)
    mean_per_classes = torch.zeros((num_classes, feature_dim), dtype=torch.float, device=features.device)
    cov_per_classes = torch.zeros((num_classes, feature_dim, feature_dim), dtype=torch.float, device=features.device) if sample_proto else None

    for idx, cls in enumerate(classes):
        mean_per_classes[idx] = torch.mean(features[indices == idx], dim=0, keepdim=False)
        if sample_proto:
            cov_per_classes[idx] = torch.matmul((features[indices == idx] - mean_per_classes[idx]).t(), (features[indices == idx] - mean_per_classes[idx]))
            if len(features[indices == idx]) == 1:
                print(f"---------- Only one sample in class {idx} ----------")
                cov_per_classes[idx] / len(features[indices == idx])
            else:
                cov_per_classes[idx] / (len(features[indices == idx]) - 1)
            # Van Ness estimator
            cov_per_classes[idx] = (1 - alpha) * cov_per_classes[idx] + alpha * torch.eye(feature_dim, device=features.device)

            # divide by the number of samples
            cov_per_classes[idx] /= len(features[indices == idx])

    return mean_per_classes, cov_per_classes
