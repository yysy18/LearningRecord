import rosbag

def list_bag_topics(bag_file):
    bag = rosbag.Bag(bag_file)
    topics = bag.get_type_and_topic_info()[1].keys()
    print("Topics in the bag file:")
    for topic in topics:
        print(topic)
    bag.close()

# 使用示例
bag_file = 'rgbd_dataset_freiburg3_large_cabinet-2hz-with-pointclouds.bag'
list_bag_topics(bag_file)
