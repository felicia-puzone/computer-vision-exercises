import torch
n=input.shape[0]
C=input.shape[1]
H=input.shape[2]
W=input.shape[3]
L=boxes[0].shape[0]
oH=output_size[0]
oW=output_size[1]
out=torch.zeros((n,L,C,oH,oW))
print(out.shape)
for i in range(n):
  print(f"evaluating {i} image of {n}")
  print(input[i,:,:,:].shape)

  box_list=torch.round(boxes[i])
  for b_i in range(L):
    y1=box_list[b_i,0]
    x1=box_list[b_i,1]
    y2=box_list[b_i,2]
    x2=box_list[b_i,3]
    print((y1,x1,y2,x2))
    for row_index in range(oW):
      for col_index in range(oH):
        H_index=[int(torch.floor(y1+(col_index*(y2-y1+1))/oH).item()),int(torch.ceil(y1+((col_index+1)*(y2-y1+1))/oH).item())]
        W_index=[int(torch.floor(x1+(row_index*(x2-x1+1))/oW).item()),int(torch.ceil(x1+((row_index+1)*(x2-x1+1))/oW).item())]
        print(H_index)
        print(W_index)
        print(input[i,:,H_index[0]:H_index[1],W_index[0]:W_index[1]].shape)
        print(torch.amax(input[i,:,H_index[0]:H_index[1],W_index[0]:W_index[1]],dim=(1,2)).shape)
        print(out[i,b_i,:,col_index,row_index].shape)
        out[i,b_i,:,col_index,row_index]=torch.amax(input[i,:,H_index[0]:H_index[1],W_index[0]:W_index[1]],dim=(1,2))
