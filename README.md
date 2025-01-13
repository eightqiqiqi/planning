# planning
![Detailed Workflow Diagram](https://github.com/user-attachments/assets/54e99af2-a213-411c-9d4a-2bbb277ed64d)

   ##Input (to RNN agent):      
   ###Input:    
   Current agent location    
   Previous action    
   Previous reward    
   Wall locations and elapsed time    
      
   ##Policy to Act:   
   ###Output:    
   The policy generates an action  based on the current state of the RNN hidden layer    
   Actions could involve moving in the environment (up, down, left, right) or invoking a rollout (simulation of future trajectories)   
      
   ##Environment to Input:   
   ###Output:    
   Updated agent location    
   Updated reward    
   New environmental states based on the action taken   
      
   ##Policy to World Model (via Think):   
   ###Output:   
   When "Think" is invoked, the policy signals the World Model to simulate potential future trajectories   
      
   ##World Model to agent:   
   ###Output:    
   The World Model predicts the likelihood of the next state reaching the goal   
