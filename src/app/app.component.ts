import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {

  constructor(private http: HttpClient) {}

  title = 'FakeNewsDetectionTFG';

  showMessage(){

    var cuadroenlace = (<HTMLInputElement>document.getElementById("cuadroenlace")).value;

    var model1 = document.querySelector('#model1:checked') !== null;
    var model2 = document.querySelector('#model2:checked') !== null;
    var model3 = document.querySelector('#model3:checked') !== null;
    var model4 = document.querySelector('#model4:checked') !== null;

    if (cuadroenlace == ""){
      return
    }
    else if (!((model1) || (model2) || (model3) || (model4))){
      return
    }
    else{

      const formData = new FormData();
      formData.append('url', cuadroenlace);
      formData.append('model1', model1.toString());
      formData.append('model2', model2.toString());
      formData.append('model3', model3.toString());
      formData.append('model4', model4.toString());

      this.http.post('http://localhost:5000/process_data', formData).subscribe(
        (response) => {
          console.log(response);
          if (response == true){
            this.showTrue(true)
            this.showFalse(false)
            this.showError(false)
          }
          else {
            this.showTrue(false)
            this.showFalse(true)
            this.showError(false)
          }
        },
        (error) => {
          this.showTrue(false)
          this.showFalse(false)
          this.showError(true)
          console.log(error);
        }
      );

    }

  }

  showTrue(cond: boolean){
    if(cond){
      var trueMessage = document.getElementById('trueMessage');
      if(trueMessage != null){
        if (trueMessage.style.visibility === '') {
          trueMessage.style.visibility = 'visible';
        } else if((trueMessage.style.visibility === 'visible')){
          trueMessage.style.visibility = 'hidden';
        }
      }
    }
  }

  showFalse(cond: boolean){
    if(cond){
      var falseMessage = document.getElementById('falseMessage');
      if(falseMessage != null){
        if (falseMessage.style.visibility === '') {
          falseMessage.style.visibility = 'visible';
        } else if((falseMessage.style.visibility === 'visible')){
          falseMessage.style.visibility = 'hidden';
        }
      }
    } 
  }

  showError(cond: boolean){
    if(cond){
      var errorMessage = document.getElementById('errorMessage');
      if(errorMessage != null){
        if (errorMessage.style.visibility === '') {
          errorMessage.style.visibility = 'visible';
        } else if((errorMessage.style.visibility === 'visible')){
          errorMessage.style.visibility = 'hidden';
        }
      }
    } 
  }

  showExplicacion(){
    var explicacion = document.getElementById('explain');
    if(explicacion != null){
      if (explicacion.style.visibility === '') {
        explicacion.style.visibility = 'visible';
      } 
      else if((explicacion.style.visibility === 'visible')){
        explicacion.style.visibility = 'hidden';
      }
      else if((explicacion.style.visibility === 'hidden')){
        explicacion.style.visibility = 'visible';
      }
    }
    
  }
  
}