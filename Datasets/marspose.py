from __future__ import division, print_function, absolute_import
import os.path as osp
import warnings
from scipy.io import loadmat

from .PoseDatasets import PoseDataset


class MarsPose(PoseDataset):
    """MARS.
    Reference:
        Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    URL: `<http://www.liangzheng.com.cn/Project/project_mars.html>`_

    Dataset statistics:
        - identities: 1261.
        - tracklets: 8298 (train) + 1980 (query) + 9330 (gallery).
        - keypoints: 17
        - cameras: 6.
    """
    #dataset_dir = 'mars'
    #dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root)) #'/home/ma1/work/PoseTrackReID'
        # self.dataset_dir = osp.join(self.root, self.dataset_dir) # '/home/ma1/work/PoseTrackReID/data/mars'
        self.dataset_dir = osp.join(self.root, '../', 'data/MARS')

        #self.download_dataset(self.dataset_dir, self.dataset_url)

        self.train_name_path = osp.join(
            self.dataset_dir, 'info/train_name.txt'
        )
        self.test_name_path = osp.join(self.dataset_dir, 'info/test_name.txt')
        self.track_train_info_path = osp.join(
            self.dataset_dir, 'info/tracks_train_info.mat'
        )
        self.track_test_info_path = osp.join(
            self.dataset_dir, 'info/tracks_test_info.mat'
        )
        self.query_IDX_path = osp.join(self.dataset_dir, 'info/query_IDX.mat')

        required_files = [
            self.dataset_dir, self.train_name_path, self.test_name_path,
            self.track_train_info_path, self.track_test_info_path,
            self.query_IDX_path
        ]
        self.check_before_run(required_files)

        train_names = self.get_names(self.train_name_path)
        test_names = self.get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path  #mat 是  8294 *4 的nndarray,  4维分别是     pid ，camid
                              )['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path
                             )['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path
                            )['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :] #按照idx找到 query的tracklets
        gallery_IDX = [
            i for i in range(track_test.shape[0]) if i not in query_IDX
        ] #除了queryidx对应的tracklets，其他的都是gallery的tracklets
        track_gallery = track_test[gallery_IDX, :]

        train = self.process_data(  #list 8298 length(tracklets) .  each tracklet is list contains keypointspath pid camid
            train_names, track_train, home_dir='bbox_train', relabel=True
        ) #长度是一个8298的list 其中每个元素是一个tuple，tuple里面是一个tracklets的信息，一个包含该tracklet的图片对应的pose的地址的list，pid，camid
        query = self.process_data(
            test_names, track_query, home_dir='bbox_test', relabel=False
        )
        gallery = self.process_data(
            test_names, track_gallery, home_dir='bbox_test', relabel=False
        )

        super(MarsPose, self).__init__(train, query, gallery, **kwargs)

    def get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def process_data(
        self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0
    ):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))  #pid 的个数，对于test 626

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)} #pid变成label，之前pid是数据集中的，是不连续的，pid2label是建立一个映射
        tracklets = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data   # 这四个也就是 mat中的4维
            if pid == -1 or pid == 0:
                continue  # junk images are just ignored, '0000' and '00-1' not
            # provided by team PoseTrack
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid] #按照上面的映射，把pid变成label
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, \
                'Error: a single tracklet contains different person images'

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, \
                'Error: images are captured under different cameras!'

            # append image names with directory information
            img_paths = [
                osp.join(self.dataset_dir, home_dir, img_name[:4], img_name) #这里在拼接路径，img_name[:4]比如是0001
                for img_name in img_names
            ]
            keypoints_paths = [
                osp.join(self.dataset_dir,
                         home_dir + '_keypoints',
                         img_name[:4],
                         img_name.replace('.jpg', '.pose')) #找到对应的tracklets下的这些图片的keypoints
                for img_name in img_names
            ]
            if len(img_paths) >= min_seq_len: # 所以如果tracklets的长度小于min_seq_len，就舍弃了这个tracklets
                img_paths = tuple(img_paths)
                tracklets.append((keypoints_paths, pid, camid))

        return tracklets

    def combine_all(self):
        warnings.warn(
            'Some query IDs do not appear in gallery. Therefore, combineall '
            'does not make any difference to Mars'
        )
