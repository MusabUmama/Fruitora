const slider = () => 
{
    const toggle1 = document.querySelector('.navi_icon');
    const navi = document.querySelector('.navi');
    const naviLinks = document.querySelectorAll('.navi li');

    toggle1.addEventListener('click', () => {
        navi.classList.toggle('navi_active');
        

        naviLinks.forEach((link,index) =>
        {
            if(link.style.animation)
            {
                link.style.animation='';
            }
            else
            {
                link.style.animation=`navLinkfade 0.5s ease forwards ${index/7+0.3}s`;
            }
        })
    })

    toggle1.addEventListener('click',()=>
    {
        toggle1.classList.toggle('burger_active');
    })
}
slider();


