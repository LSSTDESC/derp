import torch

def reg_bce_loss(output, label, binary_label, label_mask, binary_label_mask, scale_bce, test=False):
    # Regression
    reg_loss = torch.nn.SmoothL1Loss(reduction='none')(output[:, :-1], label)
    reg_loss[label_mask] = 0.0
    reduced_reg_loss = torch.mean(torch.sum(reg_loss, dim=1))
    
    # Binary classification
    bce_output = output[:, -1].reshape(-1, 1) # Take last dim for classification
    if test:
        pos_weight = None
    else:
        num_negatives = torch.sum(binary_label == 0.0, dim=0).float()
        num_positives = torch.sum(binary_label == 1.0, dim=0).float()
        #print(num_negatives, num_positives)
        pos_weight = num_negatives/(num_positives) # 1 to prevent zero division
    #print(pos_weight)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)(bce_output, binary_label)
    bce_loss[binary_label_mask] = 0.0
    reduced_bce_loss = torch.mean(bce_loss)
    
    return reduced_reg_loss + reduced_bce_loss*scale_bce