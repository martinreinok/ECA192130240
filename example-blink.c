/******************************************************************************
  RED-V_blink.c

  WRITTEN BY: Ho Yun "Bobby" Chan and "Tron Monroe"
  @ SparkFun Electronics
  DATE:  11/21/2019

  DEVELOPMENT ENVIRONMENT SPECIFICS:
    Firmware developed using Freedom Studio v4.12.0.2019-08-2
    on Windows 10

  ========== RESOURCES ==========
  Freedom E SDK

  ========== DESCRIPTION ==========
  Using the built-in LED. To test with different pin,
  simply modify the reference pin and connect a standard LED
  and 100?O resistor between the respective pin and GND.

  LICENSE: This code is released under the MIT License (http://opensource.org/licenses/MIT)
******************************************************************************/


#include <stdio.h>      //include Serial Library
#include <time.h>       //include Time library
#include <metal/gpio.h> //include GPIO library, https://sifive.github.io/freedom-metal-docs/apiref/gpio.html

//custom write delay function since we do not have one like an Arduino
void delay(int number_of_microseconds){ //not actually number of seconds

// Converting time into multiples of a hundred nS
int hundred_ns = 10 * number_of_microseconds;

// Storing start time
clock_t start_time = clock();

// looping till required time is not achieved
while (clock() < start_time + hundred_ns);

}

int main (void) {
  printf("RED-V Example: Blink\n");

  struct metal_gpio *led0; //make instance of GPIO

  //Note: The sequence of these commands matter!

  //Get gpio device handle, i.e.) define IC pin here where IC's GPIO = 5, pin silkscreen = 13
  //this is the GPIO device index that is referenced from 0, make sure to check the schematic
  led0 = metal_gpio_get_device(0);

  // quick check to see if we set the metal_gpio up correctly, this was based on the "sifive-welcome.c" example code
  if (led0 == NULL) {
    printf("LED is null.\n");
    return 1;
  }
  //Pins are set when initialized so we must disable it when we use it as an input/output
  metal_gpio_disable_input(led0, 5);

  //Set as gpio as output
  metal_gpio_enable_output(led0, 5);

  //Pins have more than one function, make sure we disconnect anything connected...
  metal_gpio_disable_pinmux(led0, 5);

  //Turn ON pin
  metal_gpio_set_pin(led0, 5, 1);


  while (1) {//loop through, sort of like an Arduino loop()

      //Turn OFF pin
      metal_gpio_set_pin(led0, 5, 0);
      //Use custom "delay" function
      delay(100000); //2000000 "micro-seconds" ~ 1 second, through experimentation...
      //Turn ON pin
      metal_gpio_set_pin(led0, 5, 1);
      //Use custom "delay" function
      delay(100000);

  }

  // return
  return 0;
}
