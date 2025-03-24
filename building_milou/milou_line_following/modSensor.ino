//-------------------------------------------------------------
/* read line sensors values 

Sensor Array 	Error Value
0 0 0 0 1	 4              
0 0 0 1 1	 3              
0 0 0 1 0	 2              
0 0 1 1 0	 1              
0 0 1 0 0	 0              
0 1 1 0 0	-1              
0 1 0 0 0	-2              
1 1 0 0 0	-3              
1 0 0 0 0	-4              

1 1 1 1 1        0 Robot found continuous line - test if an intersection or end of maze
0 0 0 0 0        0 Robot found no line: turn 180o

*/
void readLFSsensors()
{
  LFSensor[0] = digitalRead(farLeftSensor);
  LFSensor[1] = digitalRead(lineFollowSensor0);
  LFSensor[2] = digitalRead(lineFollowSensor1);
  LFSensor[3] = digitalRead(lineFollowSensor2);
  LFSensor[4] = digitalRead(lineFollowSensor3);
  LFSensor[5] = digitalRead(lineFollowSensor4);
  LFSensor[6] = digitalRead(farRightSensor);
  
  if     ((LFSensor[0]== line_reflectance )&&                                             (LFSensor[6]== line_reflectance ))  {mode = INTERSECTION; error = 0;}
  else if                                                                                 (LFSensor[6]== line_reflectance )   {mode = RIGHT_TURN; error = 0;}
  else if (LFSensor[0]== line_reflectance )                                                                                   {mode = LEFT_TURN; error = 0;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== void_reflectance )&&(LFSensor[3]== void_reflectance )&&(LFSensor[4]== void_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = NO_LINE; error = 0;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== void_reflectance )&&(LFSensor[3]== void_reflectance )&&(LFSensor[4]== void_reflectance )&&(LFSensor[5]== line_reflectance ))  {mode = FOLLOWING_LINE; error = -4;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== void_reflectance )&&(LFSensor[3]== void_reflectance )&&(LFSensor[4]== line_reflectance )&&(LFSensor[5]== line_reflectance ))  {mode = FOLLOWING_LINE; error = -3;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== void_reflectance )&&(LFSensor[3]== void_reflectance )&&(LFSensor[4]== line_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = FOLLOWING_LINE; error = -2;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== void_reflectance )&&(LFSensor[3]== line_reflectance )&&(LFSensor[4]== line_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = FOLLOWING_LINE; error = -1;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== void_reflectance )&&(LFSensor[3]== line_reflectance )&&(LFSensor[4]== void_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = FOLLOWING_LINE; error = 0;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== line_reflectance )&&(LFSensor[3]== line_reflectance )&&(LFSensor[4]== void_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = FOLLOWING_LINE; error = 1;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== line_reflectance )&&(LFSensor[3]== void_reflectance )&&(LFSensor[4]== void_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = FOLLOWING_LINE; error = 2;}
  else if((LFSensor[1]== line_reflectance )&&(LFSensor[2]== line_reflectance )&&(LFSensor[3]== void_reflectance )&&(LFSensor[4]== void_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = FOLLOWING_LINE; error = 3;}
  else if((LFSensor[1]== line_reflectance )&&(LFSensor[2]== void_reflectance )&&(LFSensor[3]== void_reflectance )&&(LFSensor[4]== void_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = FOLLOWING_LINE; error = 4;}
  else if((LFSensor[1]== line_reflectance )&&(LFSensor[2]== line_reflectance )&&(LFSensor[3]== line_reflectance )&&(LFSensor[4]== void_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = PAUSE_LINE; error = 0;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== line_reflectance )&&(LFSensor[3]== line_reflectance )&&(LFSensor[4]== line_reflectance )&&(LFSensor[5]== void_reflectance ))  {mode = PAUSE_LINE; error = 0;}
  else if((LFSensor[1]== void_reflectance )&&(LFSensor[2]== void_reflectance )&&(LFSensor[3]== line_reflectance )&&(LFSensor[4]== line_reflectance )&&(LFSensor[5]== line_reflectance ))  {mode = PAUSE_LINE; error = 0;}   
}

bool LFS_middle() {
  return digitalRead(lineFollowSensor2) == line_reflectance;
}