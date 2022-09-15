#define servoA_pin 10 //pwm pin
#define servoB_pin 3  //pwm pin
#define degree 180
byte servo[2] = {servoA_pin, servoA_pin};


void setup() {
}

void loop(){

  for(byte s=0; s<len(servo); s++){
    
    for (byte i = 0; i < degree; i++) { 
      servo_pwm(i,servo[s]);              
    }
    
    delay(100);
    
    for (byte i = degree; i > 0; i--) { 
      servo_pwm(i,servo[s]);               
    }
    
    delay(1000);
  }
}

// Custom servo motor contorl function
void servo_pwm(int x, int pin){
  int val = (x*10.25)+500;
  digitalWrite(pin,HIGH);
  delayMicroseconds(val);
  digitalWrite(pin,LOW);
  delay(10);
  }