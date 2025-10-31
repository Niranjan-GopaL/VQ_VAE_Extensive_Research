- Manual hp tuning ( so that we can be clever about what all hyperparameters to run instead of a GridSearchCV or a RandonSearchCV )
1. The first set of Configs that gave 100% codebook utilization is by the following :- 
(found while testing the effects that commitment_cost had on Codebook training )


I want you give me latex codeblock ( please write the latex within a python triple string so that I can copy paste the code )
extensive Report for the above assignment. 

I'm going to give you all my code, and output csvs. I think we made a very modular code that let us run multiple experiments with different hyperparameters in order to see how few important hp
influenced the model performance. ( don't specify this in the report, but what I'm banking on to differentiate our work from the rest is how we didn't just do what they asked us to, he went 100x more,
we focsed on understanding the architecture, asking important questions outside the assignment delivarables, we do this by performing many experiments with different `EXPERIMENT_CONFIGS` json that characeterizes the exeperiment . Maybe you can write about this methodology we have )
We essentially build an end to end pipeline that is research grade, that can help us with understanding this more deeply 


Sections ( make the Table of contents be like click to jump there, using the href module or something )
0. Methodology and Approach
    - one very big aspet is  what I specified above
    - we first modified the architecture until we got good results ( specify what constitutes a godd result exactly and numerically  )
    - then using that arhctitecture we performed different experiments
I want you to have the following sections ( not strict, we need to make sure we obey the deliverables very explicitly )
1. Architecture of VQVAE
insert an image that has Encoder
insert another image with complete vqvae architecture
2. Architecture of Prior-Training Models
    - PixelCNN
    - Attention Block Transformer

3. Discussion of VQVAE
3. Discussion of Prior Training

(for the above 2 sections we the below points are absolutely essential )
    - problems we faced, solutions we found
    - experiments we ran, which all parameters influenced the model performace.
    - ( I want you to give concerte parallels between what we theoretically know and we experimentally verified.)
    - you can leave placeholder for all the plots you might need 
    - we want to be as extensive as possible 
    - All the important questions that we found answers to, as subsubsubsection or paragraph's first sentence boldened or however you see fit
    - Things that we found wierd and didn't have any explanation for it

4. [Part C] Interpolation
All the logs and resutls for different experiments are present in the github

5. [Part C] In-painting


6. [Part C] Style Transfer
( this is still wip, you don't need to write anything here)



Feel absolutely free to use results that are not there in the reports I send. You only how to give me a list of things that you included in report, that I didn't do.
Then I'll actually do those experiments ( but I might not change your values )
Our main objective is to show how much extensively we studied and critiqued this entire project. 
We want to show unparalleled understanding and rigor in the way we conducted this study.



timestamp,data_dir,checkpoint_dir,results_dir,image_size,batch_size,num_workers,num_hiddens,num_residual_hiddens,num_residual_layers,embedding_dim,num_embeddings,commitment_cost,decay,num_epochs_vqvae,learning_rate_vqvae,min_codebook_usage,check_usage_every,experiment_name,notes,device,phase,codebook_usage_percent,active_codes,best_codebook_usage,final_perplexity,final_recon_loss,final_vq_loss,final_mse,final_psnr,final_ssim,target_achieved,training_completed

These 4 are with w.r.t to commitment cost. make commitment cost 60 severly decreased codebook utlization ( below 30%)
2025-10-02 12:46:19,./emoji_data,./checkpoints,./results,64,64,2,128,32,2,64,256,0.1,0.95,100,0.0003,50.0,5,commitment_cost_10,Decreasing commitment cost to 10,cuda,phase1,100.0,256,100.0,123.80919588529147,0.010399298073771672,0.0026509583963511083,0.011067456565797329,25.580121994018555,0.89915824,True,True
2025-10-02 12:41:10,./emoji_data,./checkpoints,./results,64,64,2,128,32,2,64,256,0.3,0.95,100,0.0003,50.0,5,commitment_cost_40,Increasing commitment cost to 40,cuda,phase1,57.8125,148,57.8125,79.49835342015976,0.011746786463146027,0.003186439671434271,0.012069420889019966,25.2037353515625,0.8812337,True,True
2025-10-02 11:26:02,./emoji_data,./checkpoints,./results,64,64,2,128,32,2,64,256,0.5,0.85,100,0.0003,50.0,5,commitment_cost_50,Increasing commitment cost to 50,cuda,phase1,50.0,128,50.0,78.45769696357922,0.012071112672296854,0.003425611499458169,0.011743423528969288,25.322650909423828,0.89728296,True,True

decay of 60 didn't get 50% codebook utilization
timestamp,data_dir,checkpoint_dir,results_dir,image_size,batch_size,num_workers,num_hiddens,num_residual_hiddens,num_residual_layers,embedding_dim,num_embeddings,commitment_cost,decay,num_epochs_vqvae,learning_rate_vqvae,min_codebook_usage,check_usage_every,experiment_name,notes,device,phase,codebook_usage_percent,active_codes,best_codebook_usage,final_perplexity,final_recon_loss,final_vq_loss,final_mse,final_psnr,final_ssim,target_achieved,training_completed
2025-10-02 10:33:38,./emoji_data,./checkpoints,./results,64,64,2,128,32,2,64,256,0.25,0.75,100,0.0003,50.0,5,decay_75,Decreasing Decay to 75,cuda,phase1,67.96875,174,67.96875,89.01722052158453,0.011844965032277962,0.0032452253851657496,0.01175544410943985,25.31821060180664,0.891961,True,True
2025-10-02 10:36:39,./emoji_data,./checkpoints,./results,64,64,2,128,32,2,64,256,0.25,0.8,100,0.0003,50.0,5,decay_88,Decreasing Decay to 80,cuda,phase1,71.09375,182,71.09375,101.30398285694612,0.011520000962683788,0.0030852721120493533,0.011715034954249859,25.33316421508789,0.8838252,True,True
2025-10-02 10:22:34,./emoji_data,./checkpoints,./results,64,64,2,128,32,2,64,256,0.25,0.85,100,0.0003,50.0,5,decay_85,Decreasing Decay Smaller Image_szie,cuda,phase1,63.28125,162,63.28125,90.62403752253606,0.011269152164459229,0.0034289341240834733,0.012460369616746902,25.065290451049805,0.89252937,True,True
2025-10-02 10:03:34,./emoji_data,./checkpoints,./results,64,64,2,128,32,2,64,256,0.25,0.95,100,0.0003,50.0,5,decay_95,Decreasing Decay Smaller Image_szie,cuda,phase1,74.21875,190,74.21875,102.16517345721905,0.011269068966309229,0.003039749792944162,0.011694756336510181,25.340688705444336,0.88683057,True,True

timestamp          , data_dir    , checkpoint_dir, results_dir, image_size, batch_size, num_workers, num_hiddens, num_residual_hiddens, num_residual_layers, embedding_dim, num_embeddings, commitment_cost, decay, num_epochs_vqvae, learning_rate_vqvae, min_codebook_usage, check_usage_every, experiment_name     , notes                                                     , device, phase , codebook_usage_percent, active_codes, best_codebook_usage, final_perplexity  , final_recon_loss    , final_vq_loss       , final_mse           , final_psnr        , final_ssim, target_achieved, training_completed
2025-10-21 11:14:07, ./emoji_data, ./checkpoints , ./results  ,         64,         64,           2,         128,                   32,                   2,            64,            256,            0.01,  0.95,              100,              0.0003,               50.0,                 5, commitment_cost_10  , Decreasing commitment cost to 10                          , cuda  , phase1,                  100.0,          256,               100.0, 133.9105707804362 , 0.010172629967713967, 0.004391347081997456, 0.010074888356029987, 25.988197326660156,  0.9032159, True           , True
2025-10-21 12:31:57, ./emoji_data, ./checkpoints , ./results  ,         64,         64,           2,         128,                   32,                   2,           256,            256,            0.01,  0.95,              100,              0.0003,               50.0,                 5, Higher_embedding_dim, "Prior isn't good enough, so lets make codebook more rich", cuda  , phase1,                  100.0,          256,               100.0, 126.86994758019081, 0.010591875164745709, 0.005640036474244717, 0.012034177780151367, 25.2164363861084  ,  0.8950758, True           , True

These are few of the experiments we did, this hopefully gives you an idea.




Please keep in mind
    Decoder is still not producing reasonable "new emogi"
    I tried sampling from a z_q that's close to known z_q ( z_q that corresponds to encoder outptu of emogiis in dataset )
    and it's really bad
    Both PixellCNN and VQVAE Decoder can reconstruct images in dataset
    but generating novel images ... I'm stuck there
keep this in mind while writing report ( like isolate this subsection or whatever,  from the rest of the report )
