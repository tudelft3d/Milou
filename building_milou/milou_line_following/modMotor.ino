
void motorStop(unsigned int time)
{
  leftServo.writeMicroseconds(1500);
  rightServo.writeMicroseconds(1500);
  delay(time);
}

void motorStop()
{
  motorStop(0);
}

//--------------------------------------------- 
void motorForward()
{
  leftServo.writeMicroseconds(1500 - (power+adjL));
  rightServo.writeMicroseconds(1500 + (power+adjR));
}

//---------------------------------------------
void motorBackward()
{
  leftServo.writeMicroseconds(1500 + power);
  rightServo.writeMicroseconds(1500 - power);
}

//---------------------------------------------
// void motorFwTime (unsigned int time)
// {
//   motorForward();
//   delay (time);
//   motorStop(200);
// }

// //---------------------------------------------
// void motorBwTime (unsigned int time)
// {
//   motorBackward();
//   delay (time);
//   motorStop(200);
// }

//------------------------------------------------
void motorTurn(int direction){
  leftServo.writeMicroseconds(1500 - (iniMotorPower+adjL)*direction);
  rightServo.writeMicroseconds(1500 - (iniMotorPower+adjR)*direction);
}

void motorTurnTimed(int direction, int degrees, float timeFactor)
{
  motorTurn(direction);
  delay(round(timeFactor*(degrees/10)));
  motorStop();
}

void motorTurnTimed(int direction, int degrees)
{
  motorTurnTimed(direction, degrees, adjTurn);
}

void motorTurnTimed180(int direction)
{
  motorTurnTimed(direction, 180, adjTurn180);
}

void motorTurnAndFindLine(int direction)
{
  motorTurn(direction);
  while(!LFS_middle()) {

  }
  motorStop();
}

void turnLeft() {
  Serial.println("PREPARE LEFT TURN");
  followLineForCarLength();
  Serial.println("TURN 20 deg");
  motorTurnTimed(LEFT, 20);
  Serial.println("TURN TO LINE");
  motorTurnAndFindLine(LEFT);
}

//---------------------------------------------------
void motorPIDcontrol()
{
  
  int leftMotorSpeed = 1500 - (iniMotorPower+adjL) - PIDvalue;
  int rightMotorSpeed = 1500 + (iniMotorPower+adjR) - PIDvalue;
  
  // The motor speed should not exceed the max PWM value
  leftMotorSpeed = constrain(leftMotorSpeed, 1000, 2000);
  rightMotorSpeed = constrain(rightMotorSpeed, 1000, 2000);
  
  leftServo.writeMicroseconds(leftMotorSpeed);
  rightServo.writeMicroseconds(rightMotorSpeed);
  
  //Serial.print (PIDvalue);
  //Serial.print (" ==> Left, Right:  ");
  //Serial.print (leftMotorSpeed);
  //Serial.print ("   ");
  //Serial.println (rightMotorSpeed);
}

//---------------------------------------------------
void motorGoForward()
{
  
  int leftMotorSpeed = 1500 - (iniMotorPower);
  int rightMotorSpeed = 1500 + (iniMotorPower);
  
  // The motor speed should not exceed the max PWM value
  // constrain(leftMotorSpeed, 1000, 2000);
  // constrain(rightMotorSpeed, 1000, 2000);
  
  leftServo.writeMicroseconds(leftMotorSpeed);
  rightServo.writeMicroseconds(rightMotorSpeed);
}

//---------------------------------------------------
void runExtraInch(void)
{
  motorPIDcontrol();
  delay(extraInch);
  // motorStop(5000);
}

//---------------------------------------------------
void runExtraHalfInch(void)
{
  motorPIDcontrol();
  delay(extraInch/2);
  // motorStop(5000);
}

//---------------------------------------------------
void runExtraQuarterInch(void)
{
  motorPIDcontrol();
  delay(extraInch/4);
  // motorStop(5000);
}

//---------------------------------------------------
void runUntilAxis(void)
{
  motorGoForward();
  delay(untilAxis);
  // motorStop(5000);
}

//---------------------------------------------------
void runExtraCentimetre(void)
{
  motorGoForward();
  delay(extraCentimetre);
}

//---------------------------------------------------
void goAndTurn(int direction, int degrees)
{
  motorPIDcontrol(); // motorGoForward
  delay(carLength);
  motorTurnTimed(direction, degrees);
}