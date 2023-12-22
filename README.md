# FINALYTICS
FINALYTICS is an AI-driven tool for curating financial insights. 
With URLs of financial articles provided, this software can scrape multiple websites at one touch, summarize the content, and act as a chatbot on the custom data.

Running the main file, the app opens in the web browser. In the app, the sidebar allows direct input of financial articles' URLs to initiate the data loading and processing. Once you click the "Click here," the system begins its operations. It first splits the text, then generates embedding vectors, and efficiently indexes them using FAISS, a powerful similarity search library.

These embeddings are stored and indexed using FAISS. The FAISS index is saved locally in a file path in pickle format, ensuring easy access, enhancing retrieval speed for future queries, and usability for subsequent use.

With the processing complete, users can then pose questions and receive answers based on the information contained within those news articles.
[This is the Screen shot of the UI.](https://github.com/jayanand100/FINALYTICS/assets/110692784/c105cee1-1d2a-4b72-af20-0038f8f8855f)

This software is optimized for the best input chunk size considering the token limit of Open AI by implementing the Map-Reduce Method. Thus a basic subscription will be enough to run this. 

