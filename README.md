CUDA_EURUSD_Pattern_eval
========================

Currency_state_evaluation

This is a CPU and GPU implementation of a simple pattern classification application which takes in 5 min EURUSD data.

Both the GPU and CPU version go through the entire data set, examines the last 'periods_back' time intervals(including the most recent 5 min period), classifys this period into (2^12) or 4096 possible pattern types(states). 

At that point it not only caches the classification for that current pattern, but also looks forward 'periods forward' steps. Then based on what happened those next periods, it calculates the overall net average price change which is associated with that particular price/volatility pattern, and stores that information in memory.

<table>
  <tr>
    <th>Number of time periods(5 min)</th><th>CPU time</th><th>GPU time</th><th>Speedup</th>
  </tr>
  <tr>
    <td>230,422</td><td>38 ms</td><td>1ms</td><td>38 x</td>
  </tr>
  
</table>
