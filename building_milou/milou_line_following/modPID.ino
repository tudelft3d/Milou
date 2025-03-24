

//--------------------------------------------------------
void calculatePID()
{
  P = error;
  I = I + error;
  D = error-previousError;
  PIDvalue = (Kp*P) + (Ki*I) + (Kd*D);
  previousError = error;
}

void followLine(void)
{
   readLFSsensors(); 
   calculatePID();
   motorPIDcontrol();   
}

void followLineForCarLength(void)
{
    startTimer = millis();  // Record the start time
    while (millis() - startTimer < carLength) {  // Run until wheel axis is aligned
        followLine();
    }
}