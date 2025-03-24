char command = 0;
char msg = 0;


void checkBTcmd()  
{ 
   msg = 0;
   while (BT1.available())   //Check if there is an available byte to read
   {
     delay(10); //Delay added to make thing stable 
     msg = BT1.read(); //Conduct a serial read
   }  
   if (msg != 0) 
   {
    //  Serial.print("Command received from BT ==> ");
    //  Serial.println(device); 
     command = msg;
     BT1.flush();
    }
}

//------------------------------------------------------------------------
void manualCmd()
{
  switch (command)
  {
    case 'g':
      mode = FOLLOWING_LINE;
      break;
    
    case 's': 
      motorStop(200); //turn off both motors
      break;

    case 'f':  
      motorForward();  
      break;

    case 'r':     
      motorTurnTimed(RIGHT, 30);
      motorStop(200);
      
      break;

   case 'l': 
      motorTurnTimed(LEFT, 30);
      motorStop(200);
      break;
    
    case 'b':  
      motorBackward();
      break;
    default:
      command = 0;
    // case 'p':
    //   Kp = command[2];
    //   break;
    
    // case 'i':
    //   Ki = command[2];
    //   break; 
    
    // case 'd':
    //   Kd = command[2];
    //   break;
  }
}