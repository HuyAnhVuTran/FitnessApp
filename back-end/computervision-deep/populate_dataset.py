import os
import shutil
from download_utils import download_images

# Define URLs
pushup_urls = [
    "https://olympustraininglab.com/wp-content/uploads/2021/01/Pushup-Core-Position.jpg",
    "https://www.muscletech.in/wp-content/uploads/2025/01/push-up-exercises.webp",
    "https://i.ytimg.com/vi/IODxDxX7oi4/sddefault.jpg",
    "https://www.verywellfit.com/thmb/QbzJaBojLh1tGjw7hI6bQZi-1tk=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/basicpushup-6d55a2fb6179471494e8fa9a04d8615a.gif",
    "https://cdn.centr.com/content/10000/9339/images/landscapewidedesktop1x-c09353b7326e788978029673753bc01a-cen-chestpress-169.jpg"
]

squat_urls = [
    "https://i.ytimg.com/vi/YaXPRqUwItQ/hqdefault.jpg",
    "https://cdn.mos.cms.futurecdn.net/hAKz2iHcz5tAamSCeNb75W.jpg",
    "https://www.womenshealthmag.com//vader-prod.hearstapps.com/images/1517/621/gettyimages-1293110207-1601051512.jpg?crop=1xw:0.7503751875937969xh;center,top",
    "https://toughmudder.co.uk/wp-content/uploads/2022/09/how-to-do-a-squat.jpg",
    "https://www.womensbest.com/cdn/shop/articles/11_Benefits_of_Squats_1200x.jpg?v=1691656885"
]

# Define directories
base_dir = "dataset"
pushup_dir = os.path.join(base_dir, "pushup")
squat_dir = os.path.join(base_dir, "squat")

# Clean up existing dummy data
print("Cleaning up old data...")
if os.path.exists(pushup_dir):
    shutil.rmtree(pushup_dir)
if os.path.exists(squat_dir):
    shutil.rmtree(squat_dir)

# Download images
print("Downloading pushup images...")
download_images(pushup_urls, pushup_dir)

print("Downloading squat images...")
download_images(squat_urls, squat_dir)

print("Dataset population complete.")
