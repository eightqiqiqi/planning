# planning
![Detailed Workflow Diagram](https://github.com/user-attachments/assets/f6ebb4e3-54f5-475f-b29e-9f5cab4b7279)

   ## Input (to RNN agent):      
   ### Input:    
   Current agent location    
   Previous action    
   Previous reward    
   Wall locations and elapsed time    
      
   ## Policy to Act:   
   ### Output:    
   The policy generates an action  based on the current state of the RNN hidden layer    
   Actions could involve moving in the environment (up, down, left, right) or invoking a rollout (simulation of future trajectories)   
      
   ## Environment to Input:   
   ### Output:    
   Updated agent location    
   Updated reward    
   New environmental states based on the action taken   
      
   ## Policy to World Model (via Think):   
   ### Output:   
   When "Think" is invoked, the policy signals the World Model to simulate potential future trajectories   
      
   ## World Model to agent:   
   ### Output:    
   The World Model predicts the likelihood of the next state reaching the goal   
