function Q = deepqlearn()
  % trains a neural network to estimate the q-function of a reinforcement
  % learning algorithm.
  % -- Sean Morrison, 2017
  
  % add the directory and all of its subdirectories to the path
  addpath(genpath('/home/seanny/Octave Scripts/blaze'))
  
  % create neural network
  TOPOLOGY = [1 10 4];
  [THETAs Xs] = nnbuild(TOPOLOGY);
  ACTFNS = cell(1,size(TOPOLOGY,2)-1);
  ACTFNS(1:end) = @sigmoid;
  
  % qlearning variables
  x=3; y=3; episodes=250;
  
  % run a qlearning batch to get the Q matrix`
  Q = qlearn(x,y,episodes)
  Q = Q/max(Q(:))
  
  % specify input and output; we want to train a regressor to map the q-function
  INPUT_TR = 1:1:x*y;
  OUTPUT_TR = Q';
  options = optimset('MaxIter', 100000);
  lambda = 0;
  
  % train the network using the q-matrix 
  THETAs = nntrain(options,THETAs,Xs,INPUT_TR,OUTPUT_TR,TOPOLOGY,ACTFNS,@msqerr,lambda);
  
  % actions -- up; down; left; right
  A = [-1; 1; -y; y];
  
  % initialize variables to begin our dqn run
  r = randi([1 x*y-1],1,episodes); S=1;
  POS = zeros(x,y); POS(S) = 1; POS(end)=2;
  imagesc(1:x,1:y,POS);
  drawnow;
  
  for n = 1:episodes
    running = true;
    while running == true
      % if landed on final square, break and go to next episode
      if S==x*y, 
        running=false; disp('Restart'); 
        break; 
      end
      
      % evaluate Q-function regression using neural net
      qvals = cell2mat(nnfeedforward(THETAs,Xs,S,ACTFNS)(end))
      m = max(qvals)
      idx = find(qvals(:)==m);
      i = idx(randi(size(idx,1)));
      
      % update position matrix
      S=S+A(i)
      POS=0*POS; POS(end)=2; POS(S)=1;
      imagesc(1:x,1:y,POS);
      drawnow;
    end
  
    % reinitialise the position of the agent to a random point on the grid
    S = r(n); POS = 0*POS; POS(S) = 1;
  end
end