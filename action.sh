logpath=$HOME/.mohou/plug_insert
rosbagpath=$logpath/rosbag

ln -sf config/* $logpath
ln -sf $rosbagpath ./
latest_rosbag_name=$(ls $rosbagpath -tl|grep train-episode|head -1|awk '{print $9}')
latest_rosbag_path=$rosbagpath/$latest_rosbag_name
echo $latest_rosbag_path
ln -sf $latest_rosbag_path $(pwd)/rosbag/latest.bag
