import torch

def kinetics_collate_fn(data):
    inputs_lst = [item['mp4'] for item in data]
    inputs = []
    num_pathways = len(inputs_lst[0])
    for pathway in range(num_pathways):
        inputs.append(
            torch.cat([item[pathway] for item in inputs_lst], dim=0)
        )
    labels = torch.cat([item['json'][0] for item in data], dim=0)
    video_idx = torch.cat([item['json'][1] for item in data], dim=0)

    return inputs, labels, video_idx


COLLATE_FN = {
    "kinetics": kinetics_collate_fn
}
