import torch
import numpy as np
import copy
from loop.loop_detect import LoopDetector
from loop.loop_rectify import LoopRectifier
from lietorch import SE3
import cv2
from vings_utils.gtsam_utils import matrix_to_tq
import os

class LoopModelGPS:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['device']['mapper']
        
        self.detector  = LoopDetector(cfg)
        self.rectifier = LoopRectifier(cfg)
        
        self.record_pair = []
        
        if self.cfg['mode'] == 'vio':
            self.camtimestamp = (np.loadtxt('/data/wuke/DATA/2024/realsense_vio/DBAF_format/camstamp.txt', dtype='str')[:, 0]).astype(np.float64)
            # self.loop_candidate_idx = [{"start_f_id": np.array([0, 1, 2]),  "end_f_id": np.array([1388, 1389, 1390])},]
            self.loop_candidate = [{"start_f_id": self.camtimestamp[np.array([0, 1, 2])],  "end_f_id": self.camtimestamp[np.array([1388, 1389, 1390])]},]
            self.loop_candidate = [{"start_f_id": self.camtimestamp[np.array([515, 516, 517])],  "end_f_id": self.camtimestamp[np.array([1100, 1101, 1102])]},]
        
        elif self.cfg['mode'] == 'vo':
            pass
        
        _ = 1
        # self.loop_candidate = [
        #     {"start_f_id": [17, 18, 19], "end_f_id": [322, 323, 324]},
        #     # {"start_f_id": [20, 21, 22], "end_f_id": [1325, 1326, 1327]},
        #     # {"start_f_id": [], "end_f_id": []},
        #     {"start_f_id": [1492, 1493, 1494], "end_f_id": [2114, 2115, 2116]},
        # ]

    
    def accept_thisloop(self, kfid_loop_start, kfid_loop_end):
        if len(self.record_pair) == 0:
            return True
        if not isinstance(kfid_loop_end, int): kfid_loop_end = kfid_loop_end.item()
        if not isinstance(kfid_loop_start, int): kfid_loop_start = kfid_loop_start.item()
            
        if kfid_loop_end - kfid_loop_start > 35:
            # 圈内的不要
            left_mask  = np.array(self.record_pair)[:, 0] < kfid_loop_start
            right_mask = np.array(self.record_pair)[:, 1] > kfid_loop_end
            if np.logical_and(left_mask, right_mask).sum() > 0: return False
            # 和之前圈距离之和小于20的不要
            MIN_ACCEPT_DISTANCE_TO_BEFORE = 20
            distance = np.abs(kfid_loop_start-np.array(self.record_pair)[:, 0]) + np.abs(kfid_loop_end-np.array(self.record_pair)[:, 1])
            if distance.min() < MIN_ACCEPT_DISTANCE_TO_BEFORE: return False
            return True    
        else:
            return False    
    
    def get_search_sorted_history_kf_index(self, history_kf_dict, curr_frame_dict):
        search_num         = self.cfg['looper']['search_num']
        loop_radius        = self.cfg['looper']['loop_radius']
        nms = 2
        distance_to_curr = torch.norm(curr_frame_dict['c2w'][:3,-1].unsqueeze(0) - history_kf_dict['c2w'][:, :3, -1], dim=-1) # (N, )
        sorted_distance_to_curr, sorted_history_kf_index = torch.sort(distance_to_curr)
        filtered_sorted_history_kf_index = sorted_history_kf_index[sorted_history_kf_index<(curr_frame_dict['idx']-loop_radius)] # This is index for curr_frame_dict.
        
        if filtered_sorted_history_kf_index.shape[0] == 0:
            return filtered_sorted_history_kf_index
        
        # normal_vec_to_curr = (history_kf_dict['c2w'][:, :3, -1] - curr_frame_dict['c2w'][:3,-1].unsqueeze(0)) # (N, 3)
        # normal_vec_to_curr = normal_vec_to_curr / (1e-4+torch.norm(normal_vec_to_curr, dim=-1, keepdim=True)) # (N, 3)
        
        # search_sorted_history_kf_index   = torch.concat([filtered_sorted_history_kf_index[:nearest_k_frame], sorted_history_kf_index[nearest_k_frame:-1:loop_search_interp]]) # (n, )
        search_sorted_history_kf_index    = torch.zeros(search_num, device=filtered_sorted_history_kf_index.device, dtype=filtered_sorted_history_kf_index.dtype)
        search_sorted_history_kf_index[0] = filtered_sorted_history_kf_index[0]
        cur_search_num = 1
        for idx in range(1, filtered_sorted_history_kf_index.shape[0]):
            curr_idx = filtered_sorted_history_kf_index[idx]
            # if  torch.abs(curr_idx - search_sorted_history_kf_index[:cur_search_num]).min() > nms and\
                # (normal_vec_to_curr[curr_idx].unsqueeze(0)*(normal_vec_to_curr[search_sorted_history_kf_index[:cur_search_num]])).sum(dim=1).max()<0.9999:
            if  torch.abs(curr_idx - search_sorted_history_kf_index[:cur_search_num]).min() > nms:
                search_sorted_history_kf_index[cur_search_num] = curr_idx
                cur_search_num += 1
            if cur_search_num >= search_num:
                break
        search_sorted_history_kf_index = search_sorted_history_kf_index[:cur_search_num]
        
        return search_sorted_history_kf_index        
    
    def detect_gps(self, gaussian_model, history_kf_dict, curr_frame_dict, debug=False):
        '''
            (1) Start Loop Detect when len(history_kf_dict)>10+loop_radius;
            (2) We actually need a global history keyframe list.
            history_kf_dict: a dict contains:{'c2w': (K, 4, 4), 'rgb': (K, 3, H, W), (Optional)'depth': (K, 1, H, W), 'viz_out_idx_to_f_idx': list}, timestamp order.
            cur_frame: single dict, {'idx': int, 'c2w': (4, 4), 'rgb': (3, H, W), 'viz_out_idx_to_f_idx': list}.
        '''
        loopdetect_dict = {}
        is_loop = False
        c2w2 = curr_frame_dict['c2w']
        
        history_f_idx = history_kf_dict['viz_out_idx_to_f_idx']
        if self.cfg['mode'] == 'vio':
            cur_f_idx     = curr_frame_dict['timestamp'].item()
        else:
            cur_f_idx     = curr_frame_dict['idx'].item()
        # print('cur_f_idx:', cur_f_idx)
        
        if self.cfg['mode'] == 'vo':
            for loop_candidate in self.loop_candidate:
                if cur_f_idx in loop_candidate['end_f_id']:
                    for hi in range(len(history_f_idx)):
                        if history_f_idx[hi] in loop_candidate['start_f_id']:
                            is_loop = True 
                            c2w1 = history_kf_dict['c2w'][hi]
                            c2w2[:3, -1] = c2w1[:3, -1]
                            c2w_loop_start2end    = c2w2.inverse() @ c2w1
                            print(c2w_loop_start2end)
                            loopdetect_dict['c2w_start2end'] = c2w_loop_start2end
                            loopdetect_dict['start_kf_idx']  = hi
                            loopdetect_dict['end_kf_idx']    = cur_f_idx
                            loopdetect_dict['pred_img1']     = None
                            self.loop_candidate.remove(loop_candidate)
                            return is_loop, loopdetect_dict
        
        elif self.cfg['mode'] == 'vio':
            for loop_candidate in self.loop_candidate:
                if np.min(np.abs(loop_candidate['end_f_id']-cur_f_idx)) < 1.0:
                    for hi in range(len(history_f_idx)):
                        if np.min(np.abs(loop_candidate['start_f_id']-history_f_idx[hi])) < 1.0:
                            is_loop = True 
                            c2w1 = history_kf_dict['c2w'][hi]
                            c2w2[:3, -1] = c2w1[:3, -1]
                            c2w_loop_start2end    = c2w2.inverse() @ c2w1
                            print(c2w_loop_start2end)
                            loopdetect_dict['c2w_start2end'] = c2w_loop_start2end
                            loopdetect_dict['start_kf_idx']  = hi
                            loopdetect_dict['end_kf_idx']    = curr_frame_dict['idx'].item()
                            loopdetect_dict['pred_img1']     = None
                            self.loop_candidate.remove(loop_candidate)
                            return is_loop, loopdetect_dict

        return False, None
            
    
    
    def calc_score_v1(self, gaussian_model, intrinsic, old_c2w_list, old_f_list):
        num_gaussians = gaussian_model._xyz.shape[0]
        _scores      = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=gaussian_model.device)
        _globalkfids = torch.zeros((num_gaussians, 3), dtype=torch.int32, device=gaussian_model.device)
        
        for curr_kf_id in range(len(old_f_list)):
            c2w = old_c2w_list[curr_kf_id]
            global_kf_id = old_f_list[curr_kf_id]
            w2c = torch.linalg.inv(c2w)
            
            # curr_scores = gaussian_model.render_scores(w2c, intrinsic) # (N, )
            rets = gaussian_model.render(w2c, intrinsic) # (N, )
            if rets['rgb'].max() < 0.1:
                gaussian_model.optimizer.zero_grad()
                continue
            
            (rets['rgb']*0).sum().backward()
            curr_scores = copy.deepcopy(gaussian_model._zeros.grad[:, 0])
            
            gaussian_model.optimizer.zero_grad()
            
            with torch.no_grad():
                curr_first_mask  = (_scores[:, 0] < curr_scores)
                curr_second_mask = (_scores[:, 0] > curr_scores) & (_scores[:, 1] < curr_scores)
                curr_third_mask  = (_scores[:, 1] > curr_scores) & (_scores[:, 2] < curr_scores)
                
                # First.
                _scores[curr_first_mask, 2] = _scores[curr_first_mask, 1].clone()
                _scores[curr_first_mask, 1] = _scores[curr_first_mask, 0].clone()
                _scores[curr_first_mask, 0] = curr_scores[curr_first_mask]
                _globalkfids[curr_first_mask, 2] = _globalkfids[curr_first_mask, 1].clone()
                _globalkfids[curr_first_mask, 1] = _globalkfids[curr_first_mask, 0].clone()
                _globalkfids[curr_first_mask, 0] = global_kf_id

                # Second.
                _scores[curr_second_mask, 2] = _scores[curr_second_mask, 1].clone()
                _scores[curr_second_mask, 1] = curr_scores[curr_second_mask]
                _globalkfids[curr_second_mask, 2] = _globalkfids[curr_second_mask, 1].clone()
                _globalkfids[curr_second_mask, 1] = global_kf_id

                # Third.
                _scores[curr_third_mask, 2] = curr_scores[curr_third_mask]
                _globalkfids[curr_third_mask, 2] = global_kf_id

        
        return _scores, _globalkfids # (N, 3), (N, 3)

    
    def calc_score_v2(self, gaussian_model, intrinsic, old_c2w_list, old_f_list):
        num_gaussians = gaussian_model._xyz.shape[0]
        _scores       = torch.ones((num_gaussians, 3), dtype=torch.float32, device=gaussian_model.device)
        # _globalkfids = torch.zeros((num_gaussians, 3), dtype=torch.int32, device=gaussian_model.device)
        _globalkfids  = gaussian_model._globalkf_id.reshape(-1, 1)
           
        return _scores, _globalkfids # (N, 3), (N, 3)
    
             
    def calc_score(self, gaussian_model, intrinsic, old_c2w_list, old_f_list):
        _scores, _globalkfids = self.calc_score_v1(gaussian_model, intrinsic, old_c2w_list, old_f_list)
        return _scores, _globalkfids


    def rectify(self, gaussian_model, tracker, history_kf_c2w, loopdetect_dict):
        '''
        history_kf_c2w: (N, 4, 4)
        loopdetect_dict = {'c2w_start2end': (4,4), 'start_kf_idx': int, 'end_kf_idx': int, 'pred_img2': (3, H, W)}
        '''
        loop_start_id, loop_end_id = int(loopdetect_dict['start_kf_idx']), int(loopdetect_dict['end_kf_idx'])
        loop_start2end_c2w         = loopdetect_dict['c2w_start2end']
        print('start2end_c2w: ', loop_start2end_c2w)
        # Rectify history_kf_list's poses using PGO.
        recitified_history_kf_c2w, rectified_scales = self.rectifier.rectify_poses(history_kf_c2w, loop_start_id, loop_end_id, loop_start2end_c2w)
        # Rectify gaussians.
        # self.rectifier.rectify_gaussians(loopdetect_dict, history_kf_c2w, recitified_history_kf_c2w, rectified_scales, self, gaussian_model)
        
        # Rectify tracker.frontend.video.poses, disps, disps_up. 
        #         tracker.frontend.video.poses_save, disps_save, disps_up_save. 
        if not self.cfg['mode'] == 'vo_nerfslam':
            # self.rectifier.rectify_tracker(gaussian_model, tracker, loop_start_id, loop_end_id, recitified_history_kf_c2w)
            # TTD 2024/12/20
            raw_loopend_c2w = SE3(tracker.frontend.video.poses_save[loop_end_id].unsqueeze(0)).inv().matrix().squeeze(0) # (4, 4)
            with torch.no_grad():
                intrinsic = {'fv': gaussian_model.tfer.fv, 'fu': gaussian_model.tfer.fu, 'cv': gaussian_model.tfer.cv, 'cu': gaussian_model.tfer.cu, 'H': gaussian_model.tfer.H, 'W': gaussian_model.tfer.W}
                for globalkf_idx in range(loop_start_id, loop_end_id+1):
                    rectified_w2c  = torch.linalg.inv(recitified_history_kf_c2w[globalkf_idx])
                    tracker.frontend.video.poses_save[globalkf_idx]    = matrix_to_tq(rectified_w2c.unsqueeze(0)).cpu().squeeze(0)
                # How stupid I am.
                new_loopend_c2w = SE3(tracker.frontend.video.poses_save[loop_end_id].unsqueeze(0)).inv().matrix().squeeze(0) # (4, 4)
                
                after_loopend_mask = (tracker.frontend.video.tstamp_save > loop_end_id)
                loopend_c2c = (new_loopend_c2w @ torch.linalg.inv(raw_loopend_c2w)).unsqueeze(0).to(torch.float32) # (1, 4, 4)
                
                tracker.frontend.video.poses_save[after_loopend_mask] = matrix_to_tq(
                    torch.linalg.inv( torch.matmul(loopend_c2c, SE3(tracker.frontend.video.poses_save[after_loopend_mask]).inv().matrix().to(torch.float32)) )
                    ).to(torch.float32)
                
                
                # TTD 2024/12/21 
                # Maybe we forget to transform poses after ?
                print("Loop start timestamp: {}, Loop end timestamp: {}".format(tracker.frontend.video.tstamp_save[loop_start_id].item(), tracker.frontend.video.tstamp_save[loop_end_id].item()))
                
                # Get and find a relative pose between "poses_save" and "poses".
                
                # Rectify tracker.frontend.video.poses, disps, disps_up. 
                # if hasattr(tracker, 'local_to_global_bias') and tracker.local_to_global_bias > 10:  
                # 应该是loop的起点不动，优化后面的;
                # 这里感觉应该;
                valid_local_num  = (tracker.frontend.video.tstamp>0).sum()
                valid_global_num = (tracker.frontend.video.tstamp_save>0).sum()
                tstamp_save_list = tracker.frontend.video.tstamp_save[:valid_global_num].to(torch.int32).tolist()
                
                print("tracker.frontend.video.tstamp: {}".format(tracker.frontend.video.tstamp[:valid_local_num].cpu().tolist()))
                print("tracker.frontend.video.tstamp_save: {}".format(tracker.frontend.video.tstamp_save[:valid_global_num].cpu().tolist()))
                
                
                cold_to_cnew_list = torch.zeros((valid_local_num, 4, 4))
                for local_id in range(valid_local_num):
                    local_tstamp = int(tracker.frontend.video.tstamp[local_id].item())
                    if local_tstamp in tstamp_save_list:
                        cur_c2w_local = SE3(tracker.frontend.video.poses[local_id].unsqueeze(0)).inv().matrix().squeeze(0)
                        cur_c2w_save  = SE3(tracker.frontend.video.poses_save[tstamp_save_list.index(local_tstamp)].unsqueeze(0)).inv().matrix().squeeze(0)
                        cold_to_cnew  = (cur_c2w_save.cpu() @ torch.linalg.inv(cur_c2w_local.cpu())).unsqueeze(0) # (4, 4)
                    if cold_to_cnew is not None:
                        cold_to_cnew_list[local_id] = cold_to_cnew
                
                # Check fisrt with out c2c.
                for local_id in range(valid_local_num):
                    if cold_to_cnew_list[local_id].sum()>0:
                        cold_to_cnew_list[:local_id] = cold_to_cnew_list[local_id].unsqueeze(0).repeat(local_id, 1, 1)
                        break
                            
                # temp_global_idx = tstamp_save_list.index(local_tstamp)
                # print(f"local_tstamp: {local_id} {local_tstamp}, save_tstamp: {temp_global_idx} {local_tstamp}")
                # print(cold_to_cnew)        
                # break
        
                all_cur_c2w_local = SE3(tracker.frontend.video.poses[:valid_local_num]).inv().matrix()
                all_cur_c2w_new   = torch.matmul(cold_to_cnew_list.cuda(), all_cur_c2w_local)
                
                tracker.frontend.video.poses[:valid_local_num] *= 0
                new_local_pose = matrix_to_tq(torch.linalg.inv(all_cur_c2w_new))
                tracker.frontend.video.poses[:valid_local_num] += new_local_pose
                
                tracker.filterx.video.poses[:valid_local_num] *= 0
                tracker.filterx.video.poses[:valid_local_num] += new_local_pose
                
                
                
                # print('🐷', (matrix_to_tq(torch.linalg.inv(all_cur_c2w_new)) - tracker.frontend.video.poses[:valid_local_num]).sum())
                
                    # local_start_id = max(0, loop_start_id - tracker.local_to_global_bias)
                    # local_end_id   = loop_end_id - tracker.local_to_global_bias
                    # DEVICE = tracker.frontend.video.poses.device
                    # for localkf_idx in range(local_start_id, local_end_id):
                    #    globalkf_idx = localkf_idx + tracker.local_to_global_bias
                    #    tracker.frontend.video.poses[localkf_idx]    = tracker.frontend.video.poses_save[globalkf_idx].to(DEVICE)
                    #    tracker.frontend.video.disps_up[localkf_idx] = tracker.frontend.video.disps_up_save[globalkf_idx].to(DEVICE)
                    #    tracker.frontend.video.disps[localkf_idx]    = tracker.frontend.video.disps_save[globalkf_idx].to(DEVICE)  
            
        else:
            self.rectifier.rectify_tracker_nerfslam(gaussian_model, tracker, loop_start_id, loop_end_id, recitified_history_kf_c2w)
            
        # Retrain Gaussians.
        # Dangerous Opiton.
        # gaussian_model.setup_optimizer()
        # self.rectifier.retrain_gaussian(gaussian_model, tracker, loop_start_id, loop_end_id)
        
        # Storage Control.
        return history_kf_c2w, recitified_history_kf_c2w, new_local_pose


    def preprocess_input(self):
        history_kf_dict = {'c2w': None, 'rgb': None}
        curr_frame_dict = {'c2w': None, 'rgb': None}
        return history_kf_dict, curr_frame_dict

        
    def run(self, gaussian_model, tracker, viz_out, idx=None, use_correct=True):
        # TODO, wuke: Decrease frame_num to be matched (<50 frames is great).
        
        new_local_pose = None
        
        '''
        history_kf_dict is {"c2w": SE3(tracker.frontend.video.images_up_save.poses_save).inv().matrix(), "rgb": tracker.frontend.video.images_up_save}.
        curr_frame_dict is {"idx": viz_out["global_kf_id"][-1], "c2w": viz_out["poses"][-1], "rgb": viz_out["images"][-1]}.
        '''
        if not self.cfg['mode'] == 'vo_nerfslam':
            last_kf = viz_out["global_kf_id"][-1]
        else:
            last_kf = torch.sum(tracker.visual_frontend.cam0_timestamps>0).item() + int(tracker.visual_frontend.cam0_timestamps[0]==0)
            
        if not self.cfg['mode'] == 'vo_nerfslam':
            history_kf_dict = {"c2w": SE3(tracker.frontend.video.poses_save[:last_kf+1]).inv().matrix().to(self.device), "rgb": tracker.frontend.video.images_up_save[:last_kf+1].to(self.device).permute(0, 3, 1, 2)}
            history_kf_dict["viz_out_idx_to_f_idx"] = tracker.frontend.video.tstamp_save[:last_kf+1].cpu().tolist()
            # "c2w": (num_global_kf, 4, 4), "rgb": (num_global_kf, 3, H, W)
        else:
            history_kf_dict = {"c2w": SE3(tracker.visual_frontend.cam0_T_world[:last_kf+1]).inv().matrix().to(self.device), "rgb": tracker.visual_frontend.cam0_images[:last_kf+1].to(self.device)/255.0}
            history_kf_dict["viz_out_idx_to_f_idx"] = tracker.visual_frontend.cam0_timestamps[:last_kf+1].cpu().tolist()
        
        '''
        history_kf_dict["rgb"]'s max value is 1.
        viz_out["images"]'s max value is 1.
        '''
        curr_frame_dict = {"idx": last_kf, "c2w": viz_out["poses"][-1], "rgb": viz_out["images"][-1].permute(2, 0, 1)[[2,1,0],...], "depth": viz_out["depths"][-1].permute(2, 0, 1)}
        curr_frame_dict["timestamp"] = viz_out["viz_out_idx_to_f_idx"][-1]
        # torch.save({"history_c2ws": history_kf_dict['c2w'], "curr_c2w": curr_frame_dict['c2w']}, "/data/wuke/workspace/VINGS-Mono/debug/loop_img.pt")
        # with torch.no_grad():
        is_loop, loopdetect_dict = self.detect_gps(gaussian_model, history_kf_dict, curr_frame_dict)

        if not self.cfg['mode'] == 'vo_nerfslam':
            poses_before = SE3(tracker.frontend.video.poses_save[:last_kf+1]).inv().matrix()
        else:
            poses_before = SE3(tracker.visual_frontend.cam0_T_world[:last_kf+1]).inv().matrix()
        
        if is_loop:
            '''
            torch.save({'history_kf_dict': history_kf_dict, 'curr_frame_dict': curr_frame_dict, 'loopdetect_dict': loopdetect_dict}, '/data/wuke/workspace/VINGS-Mono/debug/debug_loop.pt')
            '''
            print("Loop detected!")
            
            # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
            # TTD 2024/10/16, Save Tracker and Mapper.
            # Save Mathed KeyPoints.
            
            # torch.save(viz_out, self.cfg['output']['save_dir']+'/ply/viz_out_'+str(idx).zfill(5)+'.pt')
            # tracker.save_pt_ckpt("{}/ply/tracker_{}_beforecorrect.pt".format(self.cfg['output']['save_dir'], str(idx).zfill(5)))
            # gaussian_model.save_pt_ckpt("{}/ply/mapper_{}_beforecorrect.pt".format(self.cfg['output']['save_dir'], str(idx).zfill(5)))
            
            # tstamps = tracker.frontend.video.tstamp_save[:tracker.frontend.video.count_save+tracker.frontend.video.count_save_bias]
            # torch.save({"poses_before": poses_before, "poses_after": poses_after, "tstamps": tstamps, "loopdetect_dict": loopdetect_dict, "history_kf_dict": history_kf_dict}, os.path.join(self.cfg['output']['save_dir'], 'ply', 'poses_beforeafter_'+str(idx).zfill(5)+'.pt'))
            
            # TTD 2024/12/10
            # Save storage_manager's properties.
            
            # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
            # ⭐ 
            use_correct = True # False # True
            if use_correct:
                recitified_history_kf_c2w, new_xyz_world, new_local_pose = self.rectify(gaussian_model, tracker, history_kf_dict['c2w'], loopdetect_dict)
            
            if not self.cfg['mode'] == 'vo_nerfslam':
                poses_after = SE3(tracker.frontend.video.poses_save[:last_kf+1]).inv().matrix()
            else:
                poses_after = SE3(tracker.visual_frontend.cam0_T_world[:last_kf+1]).inv().matrix()
            
            # torch.save({"poses_before": poses_before, "poses_after": poses_after, "loop_start_kfid": loopdetect_dict['start_kf_idx'], "loop_end_kfid": loopdetect_dict['end_kf_idx']}, os.path.join(self.cfg['output']['save_dir'], 'ply', 'poses_beforeafter_'+str(idx).zfill(5)+'.pt'))
            
            
            # TTD 2024/10/15
            # Update *.txts.
            if not self.cfg['mode'] == 'vo_nerfslam':
                os.makedirs(os.path.join(self.cfg['output']['save_dir'], 'droid_c2w_new'), exist_ok=True)
                update_file_names = [f"{ts}.txt" for ts in tracker.frontend.video.tstamp_save[:tracker.frontend.video.count_save].cpu().tolist()]
                for idx in range(len(update_file_names)):
                    name = update_file_names[idx]
                    new_c2w = SE3(tracker.frontend.video.poses_save[idx]).inv().matrix().numpy()
                    np.savetxt(os.path.join(self.cfg['output']['save_dir'], 'droid_c2w_new', name), new_c2w)
            else:
                os.makedirs(os.path.join(self.cfg['output']['save_dir'], 'droid_c2w_new'), exist_ok=True)
                update_file_names = [f"{ts}.txt" for ts in tracker.visual_frontend.cam0_timestamps[:tracker.visual_frontend.kf_idx].cpu().tolist()]
                for idx in range(len(update_file_names)):
                    name = update_file_names[idx]
                    new_c2w = SE3(tracker.visual_frontend.cam0_T_world[idx]).cpu().inv().matrix().numpy()
                    np.savetxt(os.path.join(self.cfg['output']['save_dir'], 'droid_c2w_new', name), new_c2w)
                
        else:
            pass
        
        
        return new_local_pose
        