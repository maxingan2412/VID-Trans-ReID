import torch

def find_matching_elements(tensor1, tensor2):
    tensor1_flat = tensor1.flatten()
    tensor2_flat = tensor2.flatten()

    unique_elements1 = torch.unique(tensor1_flat)
    unique_elements2 = torch.unique(tensor2_flat)

    # 找到两个unique_elements集合中相同的元素
    common_elements = torch.tensor([elem.item() for elem in unique_elements1 if elem in unique_elements2])

    return common_elements

def compare_tensors(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    index = []
    matching_indices = []
    matchindex = []  # 新增，用于存储匹配到的元素在tensor_t中的位置

    for i in range(tensor2.size(1)):
        tensor_t = tensor2[:, i].flatten()
        matching_elements = find_matching_elements(tensor1, tensor_t)

        if matching_elements.size(0) > 0:
            matching_indices.append(matching_elements)
            index.append(i)

            # 找到匹配到的元素在tensor_t中的位置
            matching_position_in_t = torch.nonzero(tensor_t[..., None] == matching_elements, as_tuple=False)[:, 0]
            matchindex.append(matching_position_in_t)

    return matching_indices, len(matching_indices), index, matchindex

a = torch.arange(1,  4*129 * 768 + 1)
a = a.reshape(4, 129, -1)
b = a.view(1, 129, -1)
print(b.shape)

matching_indices, num_matching, index, matchindex = compare_tensors(b[:, 0], a)
# print("Matching elements:", matching_indices)
# print("Number of matching:", num_matching)
print("Indices of matching:", index)
print("Matching positions in tensor_t:", matchindex)
