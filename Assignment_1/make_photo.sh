#!/bin/bash
# Parameter 1: name of the object that we're making photos of

obj_name=$1
num_of_photos=4

for i in {1..$num_of_photos}; do
	counter=$i

	echo "4 seconds to take the photo $counter"
	sleep 4
	echo "Making photo $counter"
	raspistill -o "db/${obj_name}_${counter}.jpg"
	echo "Photo $counter made"
done
echo "All photos successfully done"
