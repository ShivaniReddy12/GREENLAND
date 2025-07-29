from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('BuyerLogin', views.BuyerLogin, name="BuyerLogin"),
	       path('BuyerLoginAction', views.BuyerLoginAction, name="BuyerLoginAction"),	   
	       path('SellerLogin', views.SellerLogin, name="SellerLogin"),
	       path('SellerLoginAction', views.SellerLoginAction, name="SellerLoginAction"),	   
	       path('Register', views.Register, name="Register"),
	       path('RegisterAction', views.RegisterAction, name="RegisterAction"),
	       path('RequestLand', views.RequestLand, name="RequestLand"),	
	       path('RequestLandAction', views.RequestLandAction, name="RequestLandAction"),
	       path('ViewRequestStatus', views.ViewRequestStatus, name="ViewRequestStatus"),
	       path('AcceptReject', views.AcceptReject, name="AcceptReject"),
	       path('AcceptRejectAction', views.AcceptRejectAction, name="AcceptRejectAction"),	
	       
]