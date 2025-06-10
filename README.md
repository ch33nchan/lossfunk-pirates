Project Write-up: How I Made an AI Create Pirates game?
So, for this project, I saw Paras's "Atari Pixels" repo and thought, "Super idea!" I wanted to see if I could take that concept and do something new with it, like using a more modern AI setup on a slightly more complex game.

Core Idea (A very simple idea, basically)
The main goal was to see if an AI could learn to make a playable video game just by watching someone else play. I didn't give it any rules. It just watched videos of the Pirates! game and had to figure out the physics and everything on its own.

My system has two main AI brains that work together:

The World Model (WorldModel_ViT): This one is the main brain. Its job is to see what's happening on the screen and predict the very next frame. Instead of a simple model, I used a Vision Transformer (ViT), which is a new-style of AI that can look at a whole sequence of frames (the last 16, in my case) to understand movement and momentum better.

The Controller (ActionToLatentMLP): Think of this as a simple translator. The World Model has its own secret language for actions; it doesn't understand "LEFT" or "RIGHT". So, I trained this small controller to take my keyboard press and translate it into the correct secret code that the World Model understands.

The Dataset (Where I got the gameplay videos)
I used a public dataset called AGAIN (Affective Game Annotations).

Source: again.institutedigitalgames.com

Why  was it so perfect?: This dataset was perfect because it didn't just have videos. It also came with detailed log files (raw_data.csv) that recorded every single key the original players pressed, with exact timings. This was super useful for training my controller model.

My Project Files (What each file does)
Here's a quick breakdown of the main scripts I put together in the src/ folder:

data_loader.py: This script is a smart one. It loads the video data piece-by-piece ("lazy loading") so my Mac's RAM wouldn't get fried. Without this, the whole thing would have just crashed.

models.py: This is the main blueprint file. It has the Python code that defines the architecture for both the World Model and the Controller.

train_world_model.py: The script I used to train the main "World Model." This is the one that takes a few hours to run because the AI has to learn all the game physics from zero.

extract_latent_codes.py: After the World Model was trained, I ran this script to create a simple mapping: for every action a player took, what was the "secret code" the AI learned for it? This creates a big json file.

train_controller.py: This script takes that json file and trains the small "Controller" model. This one is much faster to run.

play_pirates.py: This is the final result! It loads both of my trained models and lets you play the version of Pirates! that the AI is generating in real-time.

How to Run the Project (The Full Steps)
Here are the commands I ran in my terminal to get everything working. First, you have to activate the Python environment with source venv/bin/activate, and then run these one by one.

Train the World Model

Command: python3 src/train_world_model.py

What to Expect: This will take a few hours. You'll see a progress bar for the training, and it will save a checkpoint file like checkpoints/world_model_epoch_1.pth when it's done.

Extract Latent Codes

Command: python3 src/extract_latent_codes.py

What to Expect: This uses the model you just trained to create a new file: data/pirates_action_latent_pairs.json.

Train the Controller Model

Command: python3 src/train_controller.py

What to Expect: This is much faster. It will train for 20 epochs and save the best model to checkpoints/controller_best.pth.

Play the Game!

Command: python3 src/play_pirates.py

What to Expect: A game window will pop up. Just use the arrow keys and space bar to play the game that the AI is creating for you.


(PS : Don't go hard on my writing skills hehe )
