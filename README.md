# Writing Utensil Teacher AI
This project can detect five different kinds of writing utensils, which are pencils, pens, pencil cases, pencil sharpeners, and rulers. This project is helpful to young students in familiarizing with the names of the writing utensils as well as ways to strategically use it in a classroom. As I remember when I was growing up, not many students could fully, correctly use the writing utensils in the classroom. That's why I created this project so that all students can equally learn how what these utensils are and how to properly use them. 

## The Algorithm
The network used in this project is ImageNet(resnet-18). How this network works is when you give the network an image of an object, it makes a guess on what the object might be with a percentage of how accurate it thinks its guess is.

## Running this project
1. Log in to VS Code via SSH on your nano.
2. Download the 5 folders of images and the python file, imagenet.py   (Check the master branch for these)
3. Open up a new terminal.
4. Change directories to where you have all the downloaded images and the python code. (ex) if you have it under a folder called writing_utensil_teacher_AI, run the command 'cd writing_utensil_teacher_AI')
5. Run the command 'python3 imagenet.py folder of image/image' (ex) if you want to check pen_1.jpg which is located inside the pen folder, run the command 'python3 imagenet.py pen/pen_1.jpg')
   
[View a video explanation here]
[2024-08-16 00-14-44.mp4.zip](https://github.com/user-attachments/files/16632855/2024-08-16.00-14-44.mp4.zip)
or https://youtu.be/sgfsy3417ww
