Search.setIndex({docnames:["GUI_explained","crossing_detector","fingerprint_protocol_cascade","fragmentation","gallery","how_to_install","idCNN","index","modules","postprocessing","preprocessing","quickstart","requirements","validation","video_information_manager"],envversion:52,filenames:["GUI_explained.rst","crossing_detector.rst","fingerprint_protocol_cascade.rst","fragmentation.rst","gallery.rst","how_to_install.rst","idCNN.rst","index.rst","modules.rst","postprocessing.rst","preprocessing.rst","quickstart.rst","requirements.rst","validation.rst","video_information_manager.rst"],objects:{"":{accumulation_manager:[2,0,0,"-"],accumulator:[2,0,0,"-"],assign_them_all:[9,0,0,"-"],assigner:[2,0,0,"-"],blob:[10,0,0,"-"],compute_groundtruth_statistics:[13,0,0,"-"],compute_individual_groundtruth_statistics:[13,0,0,"-"],correct_impossible_velocity_jumps:[9,0,0,"-"],crossing_detector:[1,0,0,"-"],epoch_runner:[6,0,0,"-"],fragment:[3,0,0,"-"],generate_groundtruth:[13,0,0,"-"],generate_individual_groundtruth:[13,0,0,"-"],get_data:[6,0,0,"-"],get_predictions:[6,0,0,"-"],get_trajectories:[9,0,0,"-"],globalfragment:[3,0,0,"-"],id_CNN:[6,0,0,"-"],list_of_blobs:[10,0,0,"-"],list_of_fragments:[3,0,0,"-"],list_of_global_fragments:[3,0,0,"-"],model_area:[10,0,0,"-"],network_params:[6,0,0,"-"],pre_trainer:[2,0,0,"-"],segmentation:[10,0,0,"-"],stop_training_criteria:[6,0,0,"-"],store_accuracy_and_loss:[6,0,0,"-"],trainer:[2,0,0,"-"],video:[14,0,0,"-"],video_utils:[10,0,0,"-"]},"accumulation_manager.AccumulationManager":{assign_identities_to_fragments_used_for_training:[2,2,1,""],check_if_is_acceptable_for_training:[2,2,1,""],continue_accumulation:[2,3,1,""],get_P1_array_and_argsort:[2,4,1,""],get_acceptable_global_fragments_for_training:[2,2,1,""],get_images_and_labels_for_training:[2,2,1,""],get_new_images_and_labels:[2,2,1,""],is_not_certain:[2,4,1,""],p1_below_random:[2,4,1,""],reset_accumulation_variables:[2,2,1,""],reset_non_acceptable_fragment:[2,2,1,""],reset_non_acceptable_global_fragment:[2,2,1,""],set_fragment_temporary_id:[2,4,1,""],split_predictions_after_network_assignment:[2,2,1,""],update_counter:[2,2,1,""],update_fragments_used_for_training:[2,2,1,""],update_individual_fragments_used_for_training:[2,2,1,""],update_list_of_individual_fragments_used:[2,2,1,""],update_used_images_and_labels:[2,2,1,""]},"blob.Blob":{apply_model_area:[10,2,1,""],check_for_multiple_next_or_previous:[10,2,1,""],distance_from_countour_to:[10,2,1,""],get_image_for_identification:[10,2,1,""],get_nose_and_head_coordinates:[10,2,1,""],in_a_global_fragment_core:[10,2,1,""],is_a_sure_crossing:[10,2,1,""],is_a_sure_individual:[10,2,1,""],now_points_to:[10,2,1,""],overlaps_with:[10,2,1,""],set_image_for_identification:[10,2,1,""],squared_distance_to:[10,2,1,""]},"epoch_runner.EpochRunner":{next_batch:[6,2,1,""],run_epoch:[6,2,1,""]},"fragment.Fragment":{are_overlapping:[3,2,1,""],assign_identity:[3,2,1,""],check_consistency_with_coexistent_individual_fragments:[3,2,1,""],compute_P1_from_frequencies:[3,4,1,""],compute_P2_vector:[3,2,1,""],compute_border_velocity:[3,2,1,""],compute_certainty_of_individual_fragment:[3,4,1,""],compute_identification_frequencies_individual_fragment:[3,4,1,""],compute_identification_statistics:[3,2,1,""],compute_median_softmax:[3,4,1,""],frame_by_frame_velocity:[3,2,1,""],get_attribute_of_coexisting_fragments:[3,2,1,""],get_coexisting_individual_fragments_indices:[3,2,1,""],get_fixed_identities_of_coexisting_fragments:[3,2,1,""],get_missing_identities_in_coexisting_fragments:[3,2,1,""],get_neighbour_fragment:[3,2,1,""],get_possible_identities:[3,4,1,""],recompute_P2_of_coexisting_fragments:[3,2,1,""],reset:[3,2,1,""],set_P1_vector_accumulated:[3,2,1,""],set_distance_travelled:[3,2,1,""],set_partially_or_globally_accumulated:[3,2,1,""]},"get_data.DataSet":{consistency_check:[6,2,1,""],convert_labels_to_one_hot:[6,2,1,""]},"get_predictions.GetPrediction":{get_predictions_fully_connected_embedding:[6,2,1,""],get_predictions_softmax:[6,2,1,""],next_batch:[6,2,1,""]},"globalfragment.GlobalFragment":{acceptable_for_training:[3,2,1,""],check_uniqueness:[3,2,1,""],compute_start_end_frame_indices_of_individual_fragments:[3,2,1,""],get_images_and_labels_for_pretraining:[3,2,1,""],get_individual_fragments_of_global_fragment:[3,2,1,""],get_list_of_attributes_from_individual_fragments:[3,2,1,""],get_total_number_of_images:[3,2,1,""],reset:[3,2,1,""],set_candidate_for_accumulation:[3,2,1,""],set_minimum_distance_travelled:[3,2,1,""],update_individual_fragments_attribute:[3,2,1,""]},"id_CNN.ConvNetwork":{compute_batch_weights:[6,2,1,""],compute_loss_weights:[6,2,1,""],create_summaries_writers:[6,2,1,""],evaluation:[6,2,1,""],get_feed_dict:[6,2,1,""],get_fully_connected_vectors:[6,2,1,""],get_layers_to_optimize:[6,2,1,""],predict:[6,2,1,""],reinitialize_softmax_and_fully_connected:[6,2,1,""],restore:[6,2,1,""],restore_classifier:[6,2,1,""],restore_convolutional_layers:[6,2,1,""],save:[6,2,1,""],set_optimizer:[6,2,1,""],set_savers:[6,2,1,""],train:[6,2,1,""],validate:[6,2,1,""],weighted_loss:[6,2,1,""],write_summaries:[6,2,1,""]},"list_of_blobs.ListOfBlobs":{apply_model_area_to_video:[10,2,1,""],check_maximal_number_of_blob:[10,2,1,""],compute_crossing_fragment_identifier:[10,2,1,""],compute_fragment_identifier_and_blob_index:[10,2,1,""],compute_model_area_and_body_length:[10,2,1,""],compute_nose_and_head_coordinates:[10,2,1,""],compute_overlapping_between_subsequent_frames:[10,2,1,""],connect:[10,2,1,""],disconnect:[10,2,1,""],erode:[10,2,1,""],get_data_plot:[10,2,1,""],load:[10,6,1,""],reconnect:[10,2,1,""],save:[10,2,1,""],update_from_list_of_fragments:[10,2,1,""]},"list_of_fragments.ListOfFragments":{compute_P2_vectors:[3,2,1,""],compute_number_of_unique_images_used_for_pretraining:[3,2,1,""],compute_number_of_unique_images_used_for_training:[3,2,1,""],compute_ratio_of_images_used_for_pretraining:[3,2,1,""],compute_ratio_of_images_used_for_training:[3,2,1,""],compute_total_number_of_images_in_global_fragments:[3,2,1,""],create_light_list:[3,2,1,""],get_accumulable_individual_fragments_identifiers:[3,2,1,""],get_data_plot:[3,2,1,""],get_fragment_identifier_to_index_list:[3,2,1,""],get_images_from_fragments_to_assign:[3,2,1,""],get_new_images_and_labels_for_training:[3,2,1,""],get_next_fragment_to_identify:[3,2,1,""],get_not_accumulable_individual_fragments_identifiers:[3,2,1,""],get_number_of_unidentified_individual_fragments:[3,2,1,""],get_ordered_list_of_fragments:[3,2,1,""],get_stats:[3,2,1,""],load:[3,6,1,""],load_light_list:[3,2,1,""],plot_stats:[3,2,1,""],reset:[3,2,1,""],save:[3,2,1,""],save_light_list:[3,2,1,""],set_fragments_as_accumulable_or_not_accumulable:[3,2,1,""],update_fragments_dictionary:[3,2,1,""],update_from_list_of_blobs:[3,2,1,""]},"list_of_global_fragments.ListOfGlobalFragments":{compute_maximum_number_of_images:[3,2,1,""],delete_fragments_from_global_fragments:[3,2,1,""],filter_candidates_global_fragments_for_accumulation:[3,2,1,""],get_data_plot:[3,2,1,""],give_me_frequencies_first_fragment_accumulated:[3,4,1,""],load:[3,6,1,""],order_by_distance_to_the_first_global_fragment_for_accumulation:[3,2,1,""],order_by_distance_travelled:[3,2,1,""],relink_fragments_to_global_fragments:[3,2,1,""],reset:[3,2,1,""],save:[3,2,1,""],set_first_global_fragment_for_accumulation:[3,2,1,""]},"store_accuracy_and_loss.Store_Accuracy_and_Loss":{append_data:[6,2,1,""],load:[6,2,1,""],plot:[6,2,1,""],plot_global_fragments:[6,2,1,""],save:[6,2,1,""]},"video.Video":{blobs_path:[14,3,1,""],blobs_path_segmented:[14,3,1,""],check_split_video:[14,2,1,""],compute_identification_image_size:[14,2,1,""],create_accumulation_folder:[14,2,1,""],create_crossings_detector_folder:[14,2,1,""],create_embeddings_folder:[14,2,1,""],create_preprocessing_folder:[14,2,1,""],create_pretraining_folder:[14,2,1,""],create_session_folder:[14,2,1,""],create_training_folder:[14,2,1,""],create_trajectories_folder:[14,2,1,""],create_trajectories_wo_gaps_folder:[14,2,1,""],fragments_path:[14,3,1,""],get_episodes:[14,2,1,""],get_info:[14,2,1,""],global_fragments_path:[14,3,1,""],in_which_episode:[14,2,1,""],save:[14,2,1,""]},accumulation_manager:{AccumulationManager:[2,1,1,""],get_predictions_of_candidates_fragments:[2,5,1,""]},accumulator:{accumulate:[2,5,1,""],early_stop_criteria_for_accumulation:[2,5,1,""]},assign_them_all:{clean_individual_blob_before_saving:[9,5,1,""],close_trajectories_gaps:[9,5,1,""],get_forward_backward_list_of_frames:[9,5,1,""],reset_blobs_in_video_before_erosion_iteration:[9,5,1,""]},assigner:{assign:[2,5,1,""],assign_identity:[2,5,1,""],assigner:[2,5,1,""],compute_identification_statistics_for_non_accumulated_fragments:[2,5,1,""]},blob:{Blob:[10,1,1,""],full2miniframe:[10,5,1,""],remove_background_pixels:[10,5,1,""]},compute_groundtruth_statistics:{compare_tracking_against_groundtruth_no_gaps:[13,5,1,""]},correct_impossible_velocity_jumps:{compute_neighbour_fragments_and_velocities:[9,5,1,""],compute_velocities_consecutive_fragments:[9,5,1,""],correct_impossible_velocity_jumps:[9,5,1,""],correct_impossible_velocity_jumps_loop:[9,5,1,""],get_fragment_with_same_identity:[9,5,1,""],reassign:[9,5,1,""]},crossing_detector:{detect_crossings:[1,5,1,""]},epoch_runner:{EpochRunner:[6,1,1,""]},fragment:{Fragment:[3,1,1,""]},generate_groundtruth:{GroundTruthBlob:[13,1,1,""],generate_groundtruth:[13,5,1,""]},generate_individual_groundtruth:{GroundTruthBlob:[13,1,1,""],generate_individual_groundtruth:[13,5,1,""]},get_data:{DataSet:[6,1,1,""],dense_to_one_hot:[6,5,1,""],duplicate_PCA_images:[6,5,1,""],shuffle_images_and_labels:[6,5,1,""],split_data_train_and_validation:[6,5,1,""]},get_predictions:{GetPrediction:[6,1,1,""],kMeansCluster:[6,5,1,""]},get_trajectories:{assign_point_to_identity:[9,5,1,""],produce_output_dict:[9,5,1,""],produce_trajectories:[9,5,1,""]},globalfragment:{GlobalFragment:[3,1,1,""]},id_CNN:{ConvNetwork:[6,1,1,""],compute_accuracy:[6,5,1,""],compute_individual_accuracy:[6,5,1,""],createSaver:[6,5,1,""],get_checkpoint_subfolders:[6,5,1,""]},list_of_blobs:{ListOfBlobs:[10,1,1,""]},list_of_fragments:{ListOfFragments:[3,1,1,""],create_list_of_fragments:[3,5,1,""]},list_of_global_fragments:{ListOfGlobalFragments:[3,1,1,""],check_global_fragments:[3,5,1,""],create_list_of_global_fragments:[3,5,1,""],detect_beginnings:[3,5,1,""]},model_area:{ModelArea:[10,1,1,""]},network_params:{NetworkParams:[6,1,1,""]},pre_trainer:{pre_train:[2,5,1,""],pre_train_global_fragment:[2,5,1,""],pre_trainer:[2,5,1,""]},segmentation:{get_blobs_in_frame:[10,5,1,""],get_videoCapture:[10,5,1,""],resegment:[10,5,1,""],segment:[10,5,1,""],segment_episode:[10,5,1,""]},stop_training_criteria:{Stop_Training:[6,1,1,""]},store_accuracy_and_loss:{Store_Accuracy_and_Loss:[6,1,1,""]},trainer:{train:[2,5,1,""]},video:{Video:[14,1,1,""]},video_utils:{blob_extractor:[10,5,1,""],check_background_substraction:[10,5,1,""],cnt2BoundingBox:[10,5,1,""],cumpute_background:[10,5,1,""],filter_contours_by_area:[10,5,1,""],getCentroid:[10,5,1,""],get_blobs_information_per_frame:[10,5,1,""],get_bounding_box:[10,5,1,""],get_bounding_box_image:[10,5,1,""],get_pixels:[10,5,1,""],segment_frame:[10,5,1,""],sum_frames_for_bkg_per_episode_in_multiple_files_video:[10,5,1,""],sum_frames_for_bkg_per_episode_in_single_file_video:[10,5,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","staticmethod","Python static method"],"5":["py","function","Python function"],"6":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:staticmethod","5":"py:function","6":"py:classmethod"},terms:{"11g4cg3lb2yvs4ppym73yo5ilwocjlhkk":[],"128gb":5,"1tb":5,"20ghz":5,"28fp":[],"30dpf":4,"32gb":5,"3rd":2,"40ghz":5,"64bit":5,"6800k":5,"70cm":4,"7700k":5,"case":[1,2,3,6,10,11],"class":[1,2,3,6,10,13,14],"default":11,"final":[3,11,14],"float":[2,3,6,9,10,14],"function":[1,2,3,6,9,10,11],"int":[2,3,6,9,10,14],"new":[2,3,10,11],"p\u00e9rez":4,"public":[],"return":[1,2,3,6,9,10,11],"short":[1,2,3,10],"static":[2,3],"switch":11,"true":[1,2,3,6,10,13,14],"try":5,"while":[2,3,10,14],FOR:[],For:[3,4,11],Not:6,Ones:10,Such:[2,3],The:[2,3,5,6,7,9,10,11],These:[10,11],Uses:14,__call__:6,_accumulation_step:10,_blob_index:10,_cnn_model:6,_continue_accumul:2,_fc_vector:6,_fragment_identifi:10,_generated_while_closing_the_gap:10,_ident:10,_identity_corrected_closing_gap:10,_identity_corrected_solving_dupl:10,_ids_assign:3,_image_for_identif:10,_is_a_cross:10,_is_a_misclassified_individu:10,_is_an_individu:10,_is_certain:3,_is_uniqu:3,_knowledge_transfer_fold:6,_missing_id:3,_path_to_accuracy_error_data:6,_predict:2,_repeated_id:3,_restore_fold:6,_save_fold:6,_softmax_prob:2,_temporary_id:3,_used_for_train:10,_user_generated_ident:10,_was_a_cross:10,about:11,abov:[3,10],acccept:3,accept:[2,3,5,10,11],acceptable_for_train:[2,3],access:11,accord:[2,3,6,9,10,11,14],accordingli:[],account:3,accpet:2,accucaci:6,accumul:[3,6,8,10,14],accumulated_glob:3,accumulated_parti:3,accumulation_fold:3,accumulation_manag:2,accumulation_step:[3,13],accumulation_strategi:[2,3],accumulation_tri:3,accumulationmanag:2,accumulaton:2,accur:2,accuraci:[1,2,6,11],accuracy_valu:6,accurcai:8,acknowledg:[],acquir:2,activ:[0,2,5,11],adam:6,add:[],added:[2,10],addit:[2,11],adequ:7,admiss:11,adult:4,advanc:11,affect:5,after:[2,3,6,10,11,14],again:2,aim:11,algorithm:[3,5,6,10,11,14],all:[1,2,3,4,6,7,9,10,11,13],alloc:11,allow:[1,2,3,7,10,11,13],allow_partial_accumul:2,along:0,alreadi:[2,3,10,11],also:[2,6,10],ambigu:3,ambiguous_ident:3,amount:[10,11],analysi:6,ani:[3,6,9,11],anim:[2,3,4,6,7,9,10,11,13,14],anoth:9,apear:11,append:6,append_data:6,appli:[1,3,10,11,14],applic:[],apply_model_area:10,apply_model_area_to_video:10,apply_roi:14,approx:[],approxim:[1,2],are_overlap:3,area:[1,3,8,11,14],arena:4,arganda:4,argmax:[3,6],argsort:2,arrai:[2,3,6,9,10],arrang:3,arrow:11,articl:[],artifici:[3,10,11],arxiv:6,ask:5,assign:[3,8,10,11,13],assign_ident:[2,3],assign_identities_to_fragments_used_for_train:2,assign_point_to_ident:9,assign_them_al:[9,14],assigned_ident:[3,10,13],associ:[2,3,9,10],assum:[3,10],attempt:3,attr:3,attribut:[2,3,6,10,13,14],attributes_to_get:13,author:[],autom:10,automat:[2,4,11],avail:[2,3],averag:[10,11],ax_handl:6,axes_handl:6,bach:6,background:[10,11,14],bar:11,base:[2,3],bashrc:5,basic:14,batch:[2,6],batch_label:6,batch_oper:6,batch_siz:[2,6],batch_weight:6,been:[2,3,5,9,10,11,14],befor:[2,3,6,9,10,11],begin:3,behavior:4,behaviour:2,being:10,belong:[3,10,14],below:[2,3,10],bergomi:[4,7],between:[1,3,10,11],bigger:11,bkg:[10,14],blablabla:[],black:[6,10],blob:[1,3,8,9,11,13,14],blob_extractor:10,blob_hierarchy_in_starting_fram:3,blob_index:13,blobs_are_connect:10,blobs_in_episod:10,blobs_in_fram:10,blobs_in_video:[3,9,10,13],blobs_in_video_groundtruth:13,blobs_path:14,blobs_path_seg:14,bodi:[10,11,14],bool:[1,2,3,6,10,14],boolean_arrai:3,both:[1,2,3,10,14],bottom:[3,11],bound:[3,10],boundari:[2,10],bounding_box:10,bounding_box_height:10,bounding_box_imag:10,bounding_box_in_frame_coordin:[3,10],bounding_box_width:10,boundingbox:10,box:[3,10],brows:11,build:3,built:10,button:11,call:[3,10,11,14],can:[3,10,11],candidate_for_accumul:3,candidate_individual_fragments_identifi:2,cannot:[3,9],canva:2,canvas_from_gui:[2,6],cap:10,capabl:5,care:2,carri:11,cascad:[3,7,8,11],caus:[2,3],center:10,centroid:[3,9,10,13],centroid_trajectori:9,certain:[2,6,10,14],certainti:[2,3],certainty_p2:3,certainty_threshold:[2,3],certaninti:3,chang:11,chann:6,channel:[6,14],check:[2,3,4,6,9,10,11,13,14],check_background_substract:10,check_consistency_with_coexistent_individual_frag:3,check_for_loss_plateau:[2,6],check_for_multiple_next_or_previ:10,check_global_frag:3,check_if_is_acceptable_for_train:[2,3],check_maximal_number_of_blob:10,check_split_video:14,check_uniqu:3,checkpoint:6,chmod:5,chose:3,chunk:14,circular:4,cite:[],classifi:[3,10],classmethod:[3,10],clean:9,clean_individual_blob_before_sav:9,click:11,clip:10,clone:5,close:[9,10],close_trajectories_gap:9,closing_gap_stopping_criteria:9,cluster:6,cnn_model:6,cnn_models_dict:6,cnt2boundingbox:10,cnt:10,code:[5,7],coexist:[3,9],coexisting_individual_frag:3,collect:[1,2,3,4,6,7,9,10,11,14],color:6,com:[],come:4,command:5,compare_tracking_against_groundtruth_no_gap:13,complet:6,complex:[2,4],compon:3,componen:3,compos:[2,3,10],comput:[1,2,3,5,6,8,9,10,11,14],compute_accuraci:6,compute_batch_weight:6,compute_border_veloc:3,compute_certainty_of_individual_frag:3,compute_crossing_fragment_identifi:[3,10],compute_erosion_disk:9,compute_fragment_identifier_and_blob_index:[3,10],compute_groundtruth_statist:13,compute_identification_frequencies_individual_frag:3,compute_identification_image_s:14,compute_identification_statist:3,compute_identification_statistics_for_non_accumulated_frag:2,compute_individual_accuraci:6,compute_loss_weight:6,compute_maximum_number_of_imag:3,compute_median_softmax:3,compute_model_area_and_body_length:10,compute_model_veloc:9,compute_neighbour_fragments_and_veloc:9,compute_nose_and_head_coordin:10,compute_number_of_unique_images_used_for_pretrain:3,compute_number_of_unique_images_used_for_train:3,compute_overlapping_between_subsequent_fram:10,compute_p1_from_frequ:3,compute_p2_vector:3,compute_ratio_of_images_used_for_pretrain:3,compute_ratio_of_images_used_for_train:3,compute_start_end_frame_indices_of_individual_frag:3,compute_total_number_of_images_in_global_frag:3,compute_velocities_consecutive_frag:9,concaten:[3,11],concern:14,conclud:14,conda:5,condit:[3,5,6,7],connect:[6,9,10,11],consecut:[3,6,9,10],consid:[3,9,10,11],considerd:9,consist:[2,3,9],consistency_check:6,constant:6,contain:[1,2,3,6,9,10,11],content:8,continu:2,continue_accumul:2,continuum:5,contour:[10,11],contour_in_bounding_box:10,control:11,convent:11,convert:6,convert_labels_to_one_hot:6,convnetwork:[2,6],convolut:[1,2,6],coordin:10,copi:[3,6,7],copyright:[],core:[3,5,10],corner:11,correct:[4,7,8,10,11],correct_impossible_velocity_jump:9,correct_impossible_velocity_jumps_loop:9,correspod:6,correspond:[2,3,6,10,11,13],count:[3,6],counter:[2,6],cover:3,cpu:5,creat:[2,3,5,6,11,14],create_accumulation_fold:14,create_crossings_detector_fold:14,create_embeddings_fold:14,create_light_list:3,create_list_of_frag:3,create_list_of_global_frag:3,create_preprocessing_fold:14,create_pretraining_fold:14,create_session_fold:14,create_summaries_writ:6,create_training_fold:14,create_trajectories_fold:14,create_trajectories_wo_gaps_fold:14,createsav:6,creation:14,criteria:[2,8],crop:10,cross:[2,3,7,8,10,11,13,14],crossing_detector:[1,10,14],cumpute_background:10,cumsum:2,cumul:2,current:[2,6,9,10,11],cut:13,cython:12,darker:11,data:[3,8,11],data_set:6,dataset:[2,3,6,11],dcd:[7,8],deactiv:0,deal:13,debianoid:[],declar:[],dedic:5,deep:[5,7,8,10,11],deepcrossingdetector:10,defin:[10,11],definit:9,degre:6,delet:[3,11,14],delete_fragments_from_global_frag:3,dens:6,dense_to_one_hot:6,depend:[3,10,11],deriv:[],describ:[],descript:[2,3,9],desir:11,despit:11,detail:[],detect:[3,11],detect_begin:3,detect_cross:[1,3],detector:[7,8,10],develop:[4,5],deviat:10,diagon:10,dict:[3,6,9,10],dictionari:[3,6,9,10],did:5,diederik:6,differ:[10,11,13],direct:[3,9,10],disappear:11,discard:10,disconnect:10,discrimin:[10,11],discuss:[],disk:[5,11],displai:[2,11],disrimin:10,distanc:[2,3,10],distance_from_countour_to:10,distance_travel:3,distinct:10,distinguish:1,distribut:[],divid:[6,14],document:[],doe:[3,5,6,14],don:11,done:[3,9,11],doubl:11,down:11,download:[5,7],drag:[],draw:11,drive:[],dropout:6,due:5,duplic:[3,10],duplicate_pca_imag:6,dure:[2,3,6,9,10,11,14],e0154714:10,each:[1,2,3,6,9,10,11],earli:2,early_stop_criteria_for_accumul:2,easili:11,easy_instal:[],edit:[],either:[2,3,10,11],element:3,elig:3,ellips:11,ellipt:11,els:3,email:[],embed:14,empti:10,enclos:10,end:[2,3,5,6,10,13,14],end_frame_numb:9,ending_fram:[3,10],enough:[2,3],enter:11,entir:[3,6,13],env:5,environ:[5,11],episod:[10,14],episode_start_end_fram:10,episodes_start_end:[10,14],epoch:[2,8],epoch_i:6,epoch_runn:6,epochrunn:6,epochs_before_checking_stopping_condit:6,epochs_complet:6,equal:[3,10,11],equip:11,erod:[10,14],erosion_kernel_s:14,escudero:4,essenti:2,establish:2,estim:[10,11,14],estimated_body_length:10,euclidean:10,evalu:[2,6],even:[],eventu:10,everi:[2,3,6,10,11,14],everytim:14,exampl:[3,6,7],exclud:11,exclude_fc_and_softmax:6,execut:5,exist:[3,6,11,14],expect:11,explain:11,express:2,extens:11,extra:10,extract:[1,3,6,10,11],extrem:[9,13],extreme1_coordin:10,extreme2_coordin:10,facilit:3,factor:10,fail:10,fals:[1,2,3,6,10,14],far:10,faster:[5,11],fc_vector:6,fed:10,feed:3,feed_dict:6,feed_dict_train:6,feed_dict_v:6,ferrero:[4,7],figur:[1,2,4],file:[5,6,7,10,14],filter:[3,10],filter_candidates_global_fragments_for_accumul:3,filter_contours_by_area:10,final_ident:[3,10],find:[9,10],finerprint:3,fingerprint:[3,7,8],fingerprint_protocol_cascad:[],finish:2,first:[2,3,5,6,10,11],first_accumulation_flag:6,first_frame_first_global_frag:[3,9],first_global_fragment_for_accumul:3,fish:[7,10,11],fit:[],fix:3,fixed_ident:3,fixed_identity_threshold:3,flag:[2,6,10],fli:7,folder:[3,5,6,10,11,14],folder_nam:6,folder_to_save_for_paper_figur:10,follow:[2,3,5,11,13],fordward:6,form:14,format:6,found:[9,10,11],foundat:[],fps:14,fragment:[2,6,7,8,9,10,11,14],fragment_identifi:[3,13],fragment_identifier_to_index:[3,10],fragments_path:[3,14],frame:[3,9,10,11,13,14],frame_by_frame_veloc:3,frame_numb:[9,10,13,14],frame_segmented_and_mask:10,frames_per_second:9,framewis:10,francisco:[],free:11,frequenc:3,from:[1,2,3,6,7,9,10,11,14],fruit:4,fulfil:10,full2minifram:10,full:10,fullfram:10,fulli:6,func:6,functin:2,furthermor:11,futur:[3,9,10,11],gap:9,gap_interv:9,gather:[2,3],geforc:5,gener:[2,3,8,9,10,11,14],generate_groundtruth:13,generate_individual_groundtruth:13,get:[2,3,8,10,14],get_acceptable_global_fragments_for_train:2,get_accumulable_individual_fragments_identifi:3,get_attribute_of_coexisting_frag:3,get_available_and_non_available_ident:9,get_blobs_in_fram:10,get_blobs_information_per_fram:10,get_bounding_box:10,get_bounding_box_imag:10,get_candidate_identities_above_random_p2:9,get_candidate_identities_by_minimum_spe:9,get_checkpoint_subfold:6,get_coexisting_individual_fragments_indic:3,get_data:6,get_data_plot:[3,10],get_episod:14,get_feed_dict:6,get_fixed_identities_of_coexisting_frag:3,get_forward_backward_list_of_fram:9,get_fragment_identifier_to_index_list:3,get_fragment_with_same_ident:9,get_fully_connected_vector:6,get_image_for_identif:10,get_images_and_label:[],get_images_and_labels_for_pretrain:3,get_images_and_labels_for_train:2,get_images_from_fragments_to_assign:[2,3],get_individual_fragments_of_global_frag:3,get_info:14,get_layers_to_optim:6,get_list_of_attributes_from_individual_frag:3,get_missing_identities_in_coexisting_frag:3,get_neighbour_frag:3,get_new_images_and_label:2,get_new_images_and_labels_for_train:3,get_next_fragment_to_identifi:3,get_nose_and_head_coordin:10,get_not_accumulable_individual_fragments_identifi:3,get_number_of_unidentified_individual_frag:3,get_ordered_list_of_frag:3,get_p1_array_and_argsort:2,get_pixel:10,get_possible_ident:3,get_predict:6,get_predictions_fully_connected_embed:6,get_predictions_of_candidates_frag:2,get_predictions_softmax:6,get_stat:3,get_total_number_of_imag:3,get_trajectori:9,get_videocaptur:10,getcentroid:10,getpredict:[2,6],git:[5,9],git_commit:9,gitlab:[],give:[6,9,10,11],give_me_frequencies_first_fragment_accumul:3,given:[1,2,3,6,9,10,13],global:[2,6,7,8,10,13,14],global_epoch:2,global_frag:[2,3],global_fragments_path:[3,14],global_step:[2,6],globalfrag:[2,3],gmail:[],gnu:[],going:[2,6],gonzalo:[],good:[11,13],good_contour:10,good_contours_in_full_fram:10,googl:[],gpu:[2,5],grai:10,graph:[2,6,11],graphic:[7,11],greater:[3,10,11],ground:8,groundtruth:8,groundtruthblob:13,group:[5,7,13],gtx:5,gui:[2,5,7,11],gui_explain:[],gui_explained_:[],guid:0,had:[],hand:13,hard:13,has:[2,3,5,6,9,10,11,14],has_ambiguous_ident:10,has_enough_accumulated_coexisting_frag:3,has_preprocessing_paramet:14,have:[2,3,5,9,10,11],head:10,head_coordin:10,heigh:10,height:[2,3,6,10],henc:9,hera:[4,7],here:10,hierarchi:[3,10],high:[2,3],higher:[5,9,10],highli:5,hinz:[4,7],histogram:2,histori:10,hold:3,homogen:10,hong:10,hope:[],horizont:11,hot:6,how:[3,11],http:5,human:[7,8],hyperparamet:[6,11],id_cnn:6,idcnn:[2,3,8,10,11],ident:[2,3,4,6,7,9,10,11,13],identif:[2,3,4,7,8,10,11],identifi:[2,3,6,9,10,11,14],identification_image_s:10,identii:9,identitfi:10,identities_dictionary_permut:13,identitif:2,identity_corrected_closing_gap:3,identity_is_fix:3,identity_transf:2,ids:3,idtrack:[0,5,11],idtrackera:5,idtrackerai:[5,11],idtrackeraigui:[5,11],idtrackerdeep:5,ignor:10,imag:[1,2,3,6,10,11,14],image_channel:6,image_for_identif:10,image_height:6,image_s:[6,10],image_width:6,immedi:10,impli:[],imposs:8,impossible_velocity_threshold:9,in_a_global_fragment_cor:10,in_which_episod:14,includ:11,incorrect:11,index:[2,3,6,10,14],index_beginning_of_frag:3,index_individual_frag:2,index_individual_fragments_sorted_by_p1_max_to_min:2,indic:[2,3,6,10,11],indices_to_split:2,individu:[1,2,3,4,6,7,8,9,10],individual_accuraci:6,individual_accuracy_valu:6,individual_fragments_identifi:[2,3],individual_fragments_us:2,individual_fragments_used_for_train:2,indiviud:10,infer:6,inform:[2,6,7,8,9,10,11],inidividu:9,initi:[6,11],initialis:[2,3,11],input:[2,3,6,9,10],insid:[10,11],instal:[7,11],instanc:[2,3,10],instanti:[1,2,6],integ:[6,10],intel:5,intend:13,intens:[5,10,11,14],intepro:9,interest:7,interfac:[7,11],interpol:11,interpolate_trajectories_during_gap:9,intersect:[3,10],involv:2,is_a_cross:[3,13],is_a_sure_cross:10,is_a_sure_individu:10,is_an_individu:[3,13],is_certain:3,is_identifi:10,is_in_a_global_frag:3,is_knowledge_transf:6,is_not_certain:2,is_not_certain_flag:2,is_restor:6,iter:[2,3],iteration_numb:14,ith:3,its:[2,3,5,9,10,11],jimmi:6,joblib:12,join:9,jump:[8,11],juvenil:[],keep:[6,11],keep_prob:6,keep_prob_pl:6,kei:[3,9],kernel:14,keyboard:11,kind:13,kingma:6,kivi:12,kmeansclust:6,knn:6,knowledg:6,knowledge_transfer_fold:6,labe:6,label:[2,3,6,9],laboratori:7,larg:[5,7,13],last:[6,10,14],later:[3,10],latest:5,latter:[],launch:5,layer:6,layers_to_optimis:6,learn:[6,11,12],learning_r:6,least:[3,11],leav:11,left:3,legend_font_color:6,length:[2,3,6,10,11,14],librari:5,licens:[],light:[3,13],lighter:13,limit:11,line:11,link:[9,11],linux:5,list:[2,6,8,9,13,14],list_of_attribut:3,list_of_blob:[1,3,9,10,13],list_of_dictionari:3,list_of_frag:[2,3,9],list_of_global_frag:[2,3],listofblob:[1,3,9,10],listoffrag:[2,3,9],listofglobalfrag:[2,3],load:[2,3,6,9,10,11,14],load_light_list:3,locat:7,logit:6,longest:3,look:3,loop:9,loss:[1,2,8,11],loss_valu:6,loss_weights_pl:6,lower:[2,11],mac:[],made:10,main:[2,6,9],manag:[3,6,7,8],mani:[3,10,11],manual:10,map:[3,10],margin:10,mark:10,mask:10,mass:10,match:[],matplotlib:[2,12],matrix:[2,9],mattia:5,max_area:[10,14],max_num_step:6,max_number_of_blob:10,max_ratio_of_pretrained_imag:2,max_threshold:[10,14],maxima:3,maximal_images_per_anim:2,maximum:[3,10,11,14],maximum_body_length:14,maximum_number_of_blob:14,maximum_number_of_imag:3,mean:10,median:[3,10],median_body_length:10,median_softmax:3,melanogast:7,merchant:[],merg:6,meth:3,method:[2,3,4,6,9,10,13,14],min_area:[10,14],min_threshold:[10,14],miniconda2:5,miniconda:5,minimis:6,minimum:[2,3,10,11,14],minimum_number_of_frames_to_be_a_candidate_for_accumul:3,minium:3,mint18:5,mint:5,minum:3,miss:3,mistak:11,mistaken:10,model:[1,6,8,14],model_area:[1,10],modelarea:10,modifi:[3,10,11],modul:[1,2,6,10],moment:11,more:[2,3,10,11],move:9,much:9,multipl:[9,10,11],must:[7,10,11],n_class:6,name:[6,11,14],name_of_the_video_segment_numb:[],natsort:12,natur:4,ndarrai:[2,3,6,9,10,14],necessari:[2,3,11],need:[6,9,11],neighbour:3,neighbour_frag:9,neighbour_fragment_futur:9,neighbour_fragment_past:9,net:2,network:[1,2,3,5,7,8,10,11,14],network_param:6,networkparam:[2,6],neural:[1,2,5,10,11],neuron:6,never:10,new_imag:2,new_label:2,new_segmentation_threshold:10,newtrack:11,next:[3,6,10,11],next_batch:6,noisi:11,non:[2,3,10,14],non_accumulable_global_frag:3,non_consist:3,none:[2,3,6,9,10,13,14],nose:10,nose_coordin:10,note:[3,5,11],now:11,now_points_to:10,num:14,num_anim:3,num_clust:6,num_epoch:6,num_imag:6,number:[2,3,6,9,10,11,14],number_of_accumualted_individual_blob:3,number_of_accumualted_individual_frag:3,number_of_accumulable_individual_blob:3,number_of_accumulable_individual_frag:3,number_of_accumulated_imag:2,number_of_anim:[2,3,6,9,10,13,14],number_of_blob:[3,10],number_of_blobs_in_fram:10,number_of_channel:[3,6],number_of_chunk:10,number_of_class:6,number_of_coexisting_individual_frag:3,number_of_crossing_blob:3,number_of_crossing_frag:3,number_of_epochs_complet:6,number_of_frag:3,number_of_fram:[9,10],number_of_frames_for_bkg_in_episod:10,number_of_frames_in_direct:[3,9],number_of_frames_in_episod:10,number_of_global_frag:3,number_of_globally_accumulated_individual_blob:3,number_of_globally_accumulated_individual_frag:3,number_of_imag:[2,3],number_of_images_in_frag:3,number_of_images_per_individual_frag:3,number_of_individual_blob:3,number_of_individual_blobs_not_in_a_global_frag:3,number_of_individual_frag:3,number_of_individual_fragments_not_in_a_global_frag:3,number_of_not_accumulable_individual_blob:3,number_of_not_accumulable_individual_frag:3,number_of_partially_accumulated_individual_blob:3,number_of_partially_accumulated_individual_frag:3,number_of_unique_images_in_global_frag:2,numpi:12,nvida:5,nvidia:5,obejct:2,object:[1,2,3,6,9,10,11,13],obtain:[3,10],occlud:9,old_video:[2,10],onc:2,one:[3,6,9,10,11,13,14],ones:10,onli:[1,3,6,10,11],open:11,opencv:[5,10,12],oper:[3,5,6,10],opposit:11,ops_list:6,optim:6,optimis:6,optimisation_step:6,optimization_step:6,option:11,order:[0,1,2,3,10,11,14],order_by_distance_to_the_first_global_fragment_for_accumul:3,order_by_distance_travel:3,org:[],organ:0,organis:[2,3,9,10],origin:[6,10,14],original_bkg:14,original_diagon:10,original_roi:14,other:[3,6,10],otherwis:[1,3,10],out:[10,11],output:[3,6,7,9,14],outsid:[],ouytput:[],over:[2,3],overfit:6,overfitting_count:6,overlap:[3,9,10],overlaps_with:10,p1_arrai:2,p1_below_random:2,p1_below_random_flag:2,p1_vector:3,p2_vector:3,packag:5,page:[4,7],pair:10,paper:[],parallelis:[10,14],param:[2,6],paramet:[1,2,3,8,9,10,11,14],part:[2,3,5,6,9,10,11],partial:[2,3,14],particular:[6,9,10,11],particularli:2,pass:[2,6,10],past:[3,9,10,11],path:[2,3,6,9,10,14],path_to_load:3,path_to_load_blob_list_fil:10,path_to_sav:10,per:[1,2,3,10,11],perform:[2,3,6,9,10,13,14],permiss:5,permut:6,physic:9,pip:5,pixel:[3,10,11,13],pixels_coordinates_list:10,pixels_in_full_frame_ravel:10,place:[2,11],placehold:6,planar:10,plateau:2,plateu:6,pleas:[],plo:10,plot:[2,3,6,11],plot_flag:[1,2],plot_global_frag:6,plot_stat:3,point:[2,10,11],polavieja:[4,7],polavieja_lab:[],pop:11,popul:9,popup:11,poroduc:9,portion:[10,11,14],posit:[3,11],possibl:[2,3,11,13],post:[7,8],postprocess:10,potenti:10,potentially_randomly_assign:3,powershel:[],pre:1,pre_train:[2,3],pre_train_global_frag:2,preced:3,preciou:[],predicion:2,predict:[2,3,8],predictions_knn:6,prefer:13,preprint:6,preprocess:[7,8,14],present:[3,11],preserv:[3,11],press:11,pretrain:[3,6,8,14],pretrain_network_param:2,pretraining_global_frag:2,prevent:11,previou:[2,3,6,10,11],previous:[2,3,6],print:[2,6,10],print_flag:[2,6],probabl:[6,10],proce:11,procedur:[],proceed:5,process:[0,2,3,6,7,8,10,11,14],produc:[2,9,11],produce_output_dict:9,produce_trajectori:9,program:11,propag:11,properti:10,proport:6,protocol:[3,7,8,11],psutil:12,publish:[],purpos:[],put:2,pyautogui:12,pygam:12,python27:[],python2:[],python:5,pyyaml:12,quickstart:7,quit:11,r114:[],r669:[],ram:5,random:[2,3,6],rang:11,rank:2,rate:6,ratio:[2,3],ratio_new:2,ratio_old:2,ration:2,ravel:10,raw:5,reach:[2,11],read:10,readjust:[],realis:7,reassign:9,rebuild:2,receiv:[],recomend:11,recommend:[5,11],recompute_p2_of_coexisting_frag:3,reconnect:10,record:14,rectangl:[10,11],rectangular:11,recurs:[2,3],redistribut:[],reduct:14,refer:3,referenc:7,region:7,reinitialis:[2,10],reinitialize_softmax_and_fully_connect:6,reinizi:6,rel:2,relat:[2,3,6,9],relev:10,relink_fragments_to_global_frag:3,remov:[3,10],remove_background_pixel:10,repeat:3,repetit:3,repo:5,repositori:5,repres:[1,3,10,11],reprpres:3,requir:[3,7],rerio:7,reseg:10,resegmentation_paramet:14,reset:[2,3,9],reset_accumulation_vari:2,reset_blobs_in_video_before_erosion_iter:9,reset_non_acceptable_frag:2,reset_non_acceptable_global_frag:2,residu:10,resiz:14,resolut:14,resolution_reduct:14,respect:[2,3,10],respecto:9,restor:6,restore_classifi:6,restore_convolutional_lay:6,restore_fold:6,restore_folder_conv:6,restore_folder_fc_softmax:6,restore_index:6,result:3,retriev:[3,6,10,14],return_maximum_number_of_blob:10,return_store_object:1,right:[3,11],robert:[],roi:[7,10,11,14],roll_back_to:3,romero:[4,7],rotat:6,routin:2,row:[2,3],rst:[],run:[5,6],run_epoch:6,runner:8,sai:11,same:[2,3,6,9,10],sasave_summari:2,satisfi:[3,11],save:[2,3,6,9,10,11,14],save_fold:6,save_folder_conv:6,save_folder_fc_softmax:6,save_gt:13,save_light_list:3,save_summari:2,saver:6,saver_conv:6,saver_fc_softmax:6,scalar:6,scale:10,scikit:12,scope:[3,6,9],scopes_layers_to_optim:6,score:[2,3],scracth:10,script:[],seaborn:12,search:[],sec:11,second:[2,3,6,10],section:11,see:[1,2,3,6,9,10,14],segment:[3,7,8,9,14],segment_episod:10,segment_fram:10,segmentation_threshold:10,segmented_fram:10,select:[2,3,7,9,13],self:[3,10],send:6,separ:11,seri:11,serv:10,session:[2,6,7,10,14],session_newtrack:11,session_num:14,set:[1,2,3,6,9,10,11],set_candidate_for_accumul:3,set_distance_travel:3,set_first_global_fragment_for_accumul:3,set_fragment_temporary_id:2,set_fragments_as_accumulable_or_not_accumul:3,set_image_for_identif:10,set_individual_with_identity_0_as_cross:9,set_minimum_distance_travel:3,set_optim:6,set_p1_vector_accumul:3,set_partially_or_globally_accumualt:[],set_partially_or_globally_accumul:3,set_sav:6,setup:4,setuptool:[],sever:10,shallow:10,shape:[2,3,6,9,10,11],shortest:3,should:[3,6],show:11,shown:11,shuffl:6,shuffle_images_and_label:6,shuo:10,simpli:[],sinc:[3,11],singl:[1,2,3,10,11,13,14],size:[2,6,10,14],skip:11,slider:11,small:[5,7],smaller:10,softmax:[2,3,6],softmax_prob:[2,3,6],softmax_probs_median:3,softwar:[4,11],solid:11,solv:[3,10,14],some:7,soon:4,sort:[2,3],sourc:[1,2,3,5,6,9,10,11,13,14],space:[3,11],specif:[3,4,5,14],specifi:[1,3,6,14],speed:[3,5],spevif:3,split:[2,6,10,11,14],split_data_train_and_valid:6,split_predictions_after_network_assign:2,squar:[10,14],squared_distance_to:10,ssd:5,stadardis:10,stage:11,standard:10,start:[2,3,6,7,9,10,13,14],start_end:3,start_frame_numb:9,starting_epoch:6,starting_fram:[3,10],state:[2,10,11],statist:[2,3,8,9],std:10,std_toler:10,step:[2,3,5,6,7,10],stochast:6,stop:[2,8],stop_coefici:6,stop_train:6,stop_training_criteria:[2,6],store:[2,3,8,11,13,14],store_accuracy_and_error:2,store_accuracy_and_loss:[1,2,6],store_loss_and_accuraci:6,store_training_accuracy_and_loss_data:2,store_validation_accuracy_and_loss_data:2,str:[3,10,14],strategi:3,string:[2,6,9,10],structur:[],sub_folders_nam:6,sub_path:6,subfold:6,submit:[4,7],subsampl:2,subsequ:10,substiut:10,substract:10,subtract:[10,14],subtract_bkg:14,succe:5,succes:2,succesfulli:14,successfulli:11,sudo:[],suffici:2,suggest:11,sum:[2,10],sum_frames_for_bkg_per_episode_in_multiple_files_video:10,sum_frames_for_bkg_per_episode_in_single_file_video:10,summari:[1,2,3,6,10],summary_op:6,summary_str_train:6,summary_writer_train:6,summary_writer_valid:6,supplementari:4,support:5,sure:10,swim:10,symbol:11,symmetr:10,system:[5,10,11],tab:[7,11],take:[2,3],temporari:[2,3],temporarili:2,temporary_id:[2,3],tensor:6,tensorboard:[2,6],tensorflow:[2,5,6,12],term:5,termin:[2,5,10,11],test:[2,5],than:[2,3,9,10,11],thei:[3,11],them:[2,3,6,9,10,11],thi:[1,2,3,4,5,6,7,9,10,11,13],third:[2,3],those:10,thougth:[],threshold:[2,3,9,10,11,14],threshold_acceptable_accumul:2,through:[3,6],throughout:[],thw:6,time:[3,11],titan:5,titl:[],to_the_futur:[3,9],to_the_past:[3,9],toler:10,tool:[],top:3,total:[2,3,9],touch:[1,3,10,11],tqdm:12,track:[2,3,5,7,9,10,13,14],tracker:2,train:[1,2,3,8,10,11,14],trainer:8,training_dataset:6,training_flag:6,training_imag:6,training_label:6,trajectori:[8,11,14],trajectories_wo_gap:11,tranfer:2,transfer:[3,6],transform:10,travel:[2,3],trigger:2,trivial:14,truth:8,tupl:[3,6,9,10],two:[3,6,9,10,13],type:[2,3,5,9,11],typic:[6,9],under:5,unidentified_individuals_count:13,uniqu:[3,10],unmark:4,until:[2,6,11],updat:[2,3,9,10,11],update_count:2,update_fragments_dictionari:3,update_fragments_used_for_train:2,update_from_list_of_blob:3,update_from_list_of_frag:10,update_individual_fragments_attribut:3,update_individual_fragments_used_for_train:2,update_list_of_individual_fragments_us:2,update_used_images_and_label:2,upper:[],usag:7,use:[1,5,6,10,11],use_adam_optimis:6,use_network:1,use_previous_background:10,usebkg:10,used:[1,2,3,4,6,10,11,14],used_for_pretrain:3,used_for_train:[2,3,13],used_imag:2,used_label:2,useful:[],user:[3,7,10,11,13],user_generated_ident:3,uses:[5,11],using:[3,5,6,11,13],util:8,valid:[1,2,3,6,7,8,10],validated_ident:13,validation_dataset:6,validation_proport:6,valu:[1,2,3,6,9,10,11],variabl:[2,6],vector:[2,3,6],vector_valu:6,veloc:[3,8],velocities_between_frag:9,velocity_threshold:9,velocti:9,verifi:13,version:[3,5,6],vicent:4,video:[1,2,3,6,7,8,9,10,13],video_fold:14,video_information_manag:[],video_path:[6,9,10,14],video_util:10,videocaptur:10,videoname_segmentnumb:11,visibl:[2,3,10],visual:6,visualis:2,wai:[2,13],wang:10,want:11,warranti:[],was_a_cross:13,water:10,wave:10,webpag:7,weight:6,weighted_loss:6,were:11,wget:5,whatev:2,when:[2,10,11,13],where:[3,6,9,10,11,14],whether:[2,6,9,10],which:[2,3,6,10,11,13,14],whole:10,whose:[3,6,9,10,11],width:[2,3,6,10],window:[],without:3,word:11,work:[],would:[5,9],wrapper:5,write:6,write_setup:5,write_summari:6,writer:6,wrong:9,wrong_crossing_count:13,wrt:[2,3,10],www:[],x86_64:5,x_pl:6,y_logit:6,y_target_pl:6,yalp:5,yml:5,you:[5,11],your:[5,11],zebrafish:[4,10],zero:[3,10]},titles:["Graphical user interface (GUI)","DCD: Deep Crossing detector","Fingerprint protocol cascade","Fragmentation","Some videos tracked with idtracker.ai","Installation and requirements","Identification network","Welcome to idtracker.ai\u2019s documentation!","Code documentation","Post-processing","Preprocessing","Quickstart","Requirements","Human validation","Video information manager"],titleterms:{accumul:2,accurcai:6,adequ:11,area:10,assign:[2,9],blob:10,cascad:2,code:8,comput:13,content:7,copi:11,correct:9,criteria:6,cross:[1,9],data:6,dcd:1,debianoid:[],deep:1,detector:1,document:[7,8],download:11,epoch:6,exampl:11,file:11,fingerprint:2,fish:4,fli:4,fragment:3,gener:13,get:[6,9],global:[3,11],graphic:0,ground:13,groundtruth:13,group:4,gui:0,human:13,idcnn:6,identif:6,idtrack:[4,7],idtrackerai:[],imposs:9,index:7,indic:[],individu:[11,13],inform:14,instal:5,interest:11,interfac:0,jump:9,larg:4,linux:[],list:[3,10],locat:11,loss:6,mac:[],manag:[2,14],melanogast:4,model:10,modul:[],network:6,output:11,ouytput:[],paramet:6,post:9,predict:6,preprocess:[0,10,11],pretrain:2,process:9,protocol:2,quickstart:11,refer:7,region:11,requir:[5,12],rerio:4,roi:0,runner:6,search:7,segment:[10,11],select:[0,11],session:11,small:4,some:4,start:11,statist:13,step:11,stop:6,store:6,tab:0,tabl:[],track:[0,4,11],train:6,trainer:2,trajectori:9,truth:13,user:0,util:10,valid:[0,11,13],veloc:9,video:[4,11,14],welcom:[0,7],window:[]}})