# deconstructing-political-echochambers-with-recommender-systems

This project envisions an architecture for a content based recommender system that combats political echo chambers and filter bubbles by exposing them to viewpoints across the political spectrum.
This engine would serve users content topically similar to content the user interacts with that comes from the opposing point of view.  This is a sample user click and resulting recommended ad side by side generated by the system.   

![recommender_example_output](/images/engine_output.png)

-----------------------------------------------

Recommender systems tend to create what social scientists and tech activists have dubbed 'filter bubbles' - online environments where content serving algorithms feed users content that conforms to their existing beliefs while shielding users from content they may disagree with. 

![filter_bubble_illustration](/images/filter_bubble.png)

Recommenders are creating environments where many people from different political and social groups can exist on the same planet but are living in different media worlds.  

![liberal_conservative_fb](/images/liberal_conservative_fb.png)

Many believe this is at least partly to blame for an increasingly ideologically polarized society.  Filter bubbles have extremely negative consequences for democratic societies. If people cannot understand eachother, if people cannot come together and agree on basic sets of facts, then we cannot collectively solve problems. 

This project envisions a different kind of recommender system that breaks down filter bubbles rather than reinforcing them.  

To create such a system we use an LSTM Neural Network to classify text in political content as either liberal or conservative leaning, and TF-IDF vectorization with cosine similarity scoring to search for topically similar content.  The two NLP methods work together to serve content topically related to the content the user interacted with from opposite side of the political spectrum.  

![high_level_architecture](/images/high_level_structure.png)

### Data
This project used a corpus of political ads from facebook's political ad library via API. There are limits on raw data sharing allowed by facebook, so the original data will not be posted.  Using the code and your own facebook API key a user should be able to replicate this project.  This could similarly be used for political content from other ad libraries (ie twitter, google) with some modifications to the code.

## Results

The architecture can serve topically relevant

