I have a example notebook @notebooks/20260713_train_model.ipynb that has a full pipeline to train a deep learning model using pytorch, this is an example to train videos using frames and predict the type of anomaly. I want to understand the full pipeline to train, evaluate and inference the VRDFormer model using transformers. I want you to write a plan for me using only notebook to, divide it in the following steps:           

1 - Create a notebook to explore vidor dataset and data preparation with annotations                                                                                                                 
2 - Create a notebook to create the Pytorch Dataset and Dataloader and give the torch.tensor input to the model                                                                                      
3 - Create a notebook with the model train using the @models/ already with the models implemented                                                                                                    
4 - Create a notebook with the full training loop pipeline.                                                                                                                                          

- What is the data input shape tensors ?                                                                                                                                                             
- How the model understand tracking object and relations                                                                                                                                             
- the input shape to the model, torch.tensor, like, (samples, batch_size, num_frames, image_size)                                                                                                    
- The model architecture, how many layers? how many parameters?   