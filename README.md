CUDA_EURUSD_Pattern_eval
========================

__Currency_state_evaluation__


This is a CPU and GPU implementation of a simple pattern classification application which takes in 5 min EURUSD data.

Both the GPU and CPU version go through the entire data set, examines the last 'periods_back' time intervals(including the most recent 5 min period), classify this period into (2^12) or 4096 possible pattern types(states). 

At that point it not only caches the classification for that current pattern, but also looks forward 'periods forward' steps. Then based on what happened those next periods, it calculates the overall net average price change which is associated with that particular price/volatility pattern, and stores that information in memory.  

So essentially the code 'trains' on the past data, then that cached information can be used to evaluate the current state, and based on previous data predict the expected price change for the next 4-interval period.  


This version demonstrates the fast and easy use of Atomic functions, which have been significanly improved in the newer Nvidia Kepler class GPUs. It also demonstrates the increased speed of Global memory access, which has also been improved in this new generation of Nvidia GPUs.
 
 ____

<table>
  <tr>
    <th>Number of time periods(5 min)</th><th>Periods back</th><th>Periods forward</th><th>CPU time</th><th>GPU time</th><th>Speedup</th>
  </tr>
  <tr>
    <td>230,422</td><td>9 </td><td>4 </td><td>38 ms</td><td>1ms</td><td>38 x</td>
  </tr>
  
</table>

  ___

While this is not a task which is ideal for the GPU, it still is able to fully process and fill in all of the 5-min data for a 4 year period in less than 1 ms, compared to about 38 ms for the equivalent CPU version.

Since HFT trading is very dependant on the speed of the application, a more robust version of this prototype CUDA kernel could be used to output a future expected price change based on the most current state, and the history associated with this state.

In this version 12 inputs are used to describe the state, and the application could be adjusted to use more custom inputs and determine which are the most predictive. 

The code does display the most predictive inputs for bullish and bearish price changes(based on past data), and also estimates the expected price change for each state.


__project hardware/software configuration:__

Language: C++ using CUDA and the CUDA 5.0 SDK. Memory management is in C style

CPU used: Intel Core I-7 3770 3.5 ghz, 3.9 ghz target

GPU used: Nvidia Tesla K20 5GB

motherboard: MAXIMUS V GENE, PCI-e 3.0
